import io
import traceback
import contextlib

def execute_code_blocks(code: str):
    code_blocks = code.split('\n\n')
    exec_globals = {}
    executed_blocks = []

    for i, block in enumerate(code_blocks):
        cleaned = block.strip()
        if not cleaned or any(keyword in cleaned.lower() for keyword in ["sure!", "make sure", "example", "```"]):
            continue

        stdout_buffer = io.StringIO()
        output = ""
        try:
            with contextlib.redirect_stdout(stdout_buffer):
                exec(cleaned, exec_globals)
            output = stdout_buffer.getvalue().strip() or "Executed successfully."
        except Exception:
            output = traceback.format_exc()

        executed_blocks.append({
            "cell": i + 1,
            "code": cleaned,
            "output": output
        })

    return executed_blocks
