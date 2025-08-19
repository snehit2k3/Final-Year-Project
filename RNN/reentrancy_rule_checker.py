import re

def check_external_before_state_update(code):
    """
    Detect if any external call is made before state variable update
    """

    # Normalize lines
    code_lines = code.splitlines()
    external_call_lines = []
    state_update_lines = []

    for i, line in enumerate(code_lines):
        # Remove comments and extra spaces
        line_clean = re.sub(r"//.*", "", line).strip().lower().replace(" ", "")

        # External interaction check
        if (
            ".call{" in line_clean or
            ".call(" in line_clean or
            ".transfer(" in line_clean or
            ".send(" in line_clean or
            ".delegatecall(" in line_clean
        ):
            external_call_lines.append(i)

        # Check for state update to common vars like balances, mapping, etc.
        if re.search(r"userbalances\[.*?\]\s*[-+]?=", line_clean):
            state_update_lines.append(i)

    if not external_call_lines:
        return False  # No external call found

    # Check if any external call occurs before state update
    for ext_line in external_call_lines:
        for state_line in state_update_lines:
            if ext_line < state_line:
                return True  # External call before state update

    return False
