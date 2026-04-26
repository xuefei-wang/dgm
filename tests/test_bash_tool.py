import asyncio

import pytest
from tools import bash as bash_module
from tools.bash import tool_function, BashSession

@pytest.fixture
def bash_session():
    """Create a BashSession instance for testing."""
    session = BashSession()
    return session

class TestBashTool:
    def test_simple_command(self):
        """Test running a simple command."""
        result = tool_function("echo 'hello world'")
        assert "hello world" in result

    def test_multiple_commands(self):
        """Test running multiple commands in sequence."""
        result = tool_function("echo 'first' && echo 'second'")
        assert "first" in result
        assert "second" in result

    def test_command_with_error(self):
        """Test running a command that produces an error."""
        result = tool_function("ls /nonexistent/directory")
        assert "Error" in result
        assert "No such file or directory" in result

    def test_environment_variables(self):
        """Test command with environment variables."""
        result = tool_function("TEST_VAR='hello' && echo $TEST_VAR")
        assert "hello" in result

    def test_command_output_processing(self):
        """Test processing of command output."""
        commands = [
            "echo 'line1'",
            "echo 'line2'",
            "echo 'line3'"
        ]
        result = tool_function(" && ".join(commands))
        assert all(f"line{i}" in result for i in range(1, 4))

    def test_long_running_command(self):
        """Test behavior with a long-running command."""
        result = tool_function("sleep 1 && echo 'done'")
        assert "done" in result

    @pytest.mark.parametrize("invalid_command", [
        "invalid_command_name",
        "cd /nonexistent/path",
        "/bin/nonexistent"
    ])
    def test_invalid_commands(self, invalid_command):
        """Test various invalid commands."""
        result = tool_function(invalid_command)
        assert "Error" in result or "command not found" in result

    def test_command_with_special_chars(self):
        """Test command with special characters."""
        result = tool_function("echo 'test with spaces and !@#$%^&*()'")
        assert "test with spaces" in result
        assert "!@#$%^&*()" in result

    def test_multiple_line_output(self):
        """Test handling of multiple line output."""
        command = """printf 'line1\nline2\nline3'"""
        result = tool_function(command)
        assert "line1" in result
        assert "line2" in result
        assert "line3" in result

    def test_large_output_handling(self):
        """Test handling of large command output."""
        # Generate a large output
        command = "for i in {1..100}; do echo \"Line $i\"; done"
        result = tool_function(command)
        assert "Line 1" in result
        assert "Line 100" in result

    def test_real_session_stop_kills_process_group(self):
        """Regression guard for the actual subprocess teardown: stop()
        must reap not just the bash leader but also any backgrounded
        children (per the tool docstring inviting `sleep 30 &`). This
        complements the FakeSession test below — that one cannot detect
        process-group leaks because it bypasses the OS entirely."""
        import os
        import signal

        async def _exercise():
            session = bash_module.BashSession()
            await session.start()
            bash_pid = session._process.pid
            # Spawn a long-running child INSIDE the bash session and capture
            # its PID. Without process-group cleanup in stop(), this child
            # would be reparented to init and survive past stop().
            output, _ = await session.run("sleep 30 & echo $!")
            try:
                child_pid = int(output.strip().splitlines()[-1])
            except (ValueError, IndexError) as exc:
                raise AssertionError(f"could not parse child PID from {output!r}") from exc

            assert os.path.exists(f"/proc/{bash_pid}"), "bash leader should be alive before stop"
            assert os.path.exists(f"/proc/{child_pid}"), "child should be alive before stop"

            await session.stop()

            # Give the OS a brief moment to deliver the signal and reap.
            for _ in range(20):
                if not os.path.exists(f"/proc/{child_pid}"):
                    break
                await asyncio.sleep(0.05)

            assert not os.path.exists(f"/proc/{bash_pid}"), "bash leader leaked past stop()"
            child_alive = os.path.exists(f"/proc/{child_pid}")
            if child_alive:
                # Be a good citizen: clean up the leak so it doesn't pollute
                # the test runner.
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                raise AssertionError(
                    f"backgrounded child {child_pid} survived stop() — "
                    "process-group cleanup regression"
                )

        asyncio.run(_exercise())

    def test_tool_function_call_stops_session(self, monkeypatch):
        class FakeSession:
            last_instance = None

            def __init__(self):
                FakeSession.last_instance = self
                self._started = False
                self.stop_called = False

            async def start(self):
                self._started = True

            async def run(self, command):
                assert command == "echo test"
                return "test", ""

            async def stop(self):
                self.stop_called = True
                self._started = False

        monkeypatch.setattr(bash_module, "BashSession", FakeSession)

        result = asyncio.run(bash_module.tool_function_call("echo test"))

        assert result == "test"
        assert FakeSession.last_instance is not None
        assert FakeSession.last_instance.stop_called is True
