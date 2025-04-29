from datetime import time

def label_session(ts):
    """Helper to label trading session (UTC time) including overlaps."""
    if time(0, 0) <= ts.time() < time(7, 0):
        return 'Asia'
    elif time(7, 0) <= ts.time() < time(8, 0):
        return 'Asia + London Overlap'
    elif time(8, 0) <= ts.time() < time(13, 0):
        return 'London'
    elif time(13, 0) <= ts.time() < time(15, 0):
        return 'London + NY Overlap'
    elif time(15, 0) <= ts.time() < time(20, 0):
        return 'New York'
    else:
        return 'Other'
