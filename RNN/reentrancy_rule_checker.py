import re

def find_mapping_names(code):
    # Matches: mapping(address => uint) public balances;
    mapping_pattern = re.compile(r'mapping\s*\([^;{]+?\)\s*(?:public|private|internal|external)?\s*([A-Za-z_]\w*)', re.IGNORECASE)
    return set(mapping_pattern.findall(code))

def extract_functions(code):
    """
    Extract function bodies using brace-matching. Returns list of (signature, body_text, start_line)
    """
    functions = []
    # find 'function' keywords and then capture braces
    for m in re.finditer(r'\bfunction\b', code):
        start = m.start()
        # find the next '{' after the function keyword
        brace_open = code.find('{', start)
        if brace_open == -1:
            continue
        # Now match braces
        i = brace_open
        depth = 0
        while i < len(code):
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
                if depth == 0:
                    body = code[brace_open+1:i]
                    # get signature (text between 'function' and first '{')
                    sig = code[start:brace_open].strip()
                    # compute start line for reporting
                    start_line = code.count('\n', 0, start) + 1
                    functions.append((sig, body, start_line))
                    break
            i += 1
    return functions

def line_index_in_code(code, offset):
    """Helper: get 0-based line index for a character offset"""
    return code.count('\n', 0, offset)

def detect_external_before_state_update(solidity_code):
    code = solidity_code
    # 1) detect candidate mapping/state names
    mapping_names = find_mapping_names(code)
    # Add some common state variable keywords that often store balances
    common_names = {'balance', 'balances', 'deposit', 'deposits', 'vault', 'funds'}
    mapping_names.update(common_names)

    # 2) extract functions and analyze per-function
    functions = extract_functions(code)
    external_call_regex = re.compile(
        r'(\.call\b|\.send\b|\.transfer\b|delegatecall\b|call\.value\b|\w+\s*\.\s*\w+\s*\()', re.IGNORECASE
    )
    # state update: assignment ( = ), increment/decrement ( +=, -= ), or set to 0
    # specifically look for updates to mapping-like identifiers, e.g. balances[msg.sender] = ...
    state_update_patterns = []
    for name in mapping_names:
        # variations: name[...], name = ..., name += ..., name -= ...
        state_update_patterns.append(re.compile(r'\b' + re.escape(name) + r'\s*\[.*?\]\s*(?:=|\+=|-=)', re.IGNORECASE | re.DOTALL))
        state_update_patterns.append(re.compile(r'\b' + re.escape(name) + r'\s*(?:=|\+=|-=)', re.IGNORECASE))
    # fallback generic pattern: any assignment that looks like changing state (risky)
    generic_state_assign = re.compile(r'\b([A-Za-z_]\w*)\s*(?:=|\+=|-=)\s*[^;]+;', re.IGNORECASE)

    for sig, body, start_line in functions:
        # split body into lines for ordering
        # remove inline comments but keep code structure
        clean_body = re.sub(r'//.*', '', body)
        # keep multi-line statements intact; we will work with an index of line offsets
        lines = clean_body.splitlines()
        # Build a list of (line_number_in_file, line_text)
        line_offsets = []
        # compute base offset of body in code for line numbers
        # find the offset of the body start (first occurrence of body after signature) - approximate
        body_offset = code.find(body)
        for idx, ln in enumerate(lines):
            # global line number
            global_ln = (code[:body_offset].count('\n') if body_offset!=-1 else 0) + idx + 1
            line_offsets.append((global_ln, ln.strip()))

        # scan in order
        saw_state_update = False
        # store the first state update line number (if any)
        first_state_line_no = None
        # store first external call line number
        first_external_line_no = None

        for ln_no, ln in line_offsets:
            text = ln
            # ignore empty lines
            if not text:
                continue
            # check for external call
            if external_call_regex.search(text):
                if first_external_line_no is None:
                    first_external_line_no = ln_no

            # check for state updates to known names
            matched_state = False
            for pat in state_update_patterns:
                if pat.search(text):
                    matched_state = True
                    break
            if not matched_state:
                # check generic assignment - could be state update but may be local var
                if generic_state_assign.search(text):
                    matched_state = True
            if matched_state and first_state_line_no is None:
                first_state_line_no = ln_no

            # Quick decision: if external seen and state not yet seen -> vulnerability
            if first_external_line_no is not None and (first_state_line_no is None or first_external_line_no < first_state_line_no):
                # External call occurs before any state update in this function
                return True

        # if reached end of function and no external before state update, continue to next function

    # No function showed external call before state update
    return False
