def encode_role(role):

    roles = {
        "Batsman":0,
        "Bowler":1,
        "Wicketkeeper":2,
        "All-Rounder":3
    }

    return roles[role]


def encode_format(fmt):

    formats = {
        "T20":0,
        "ODI":1,
        "Test":2
    }

    return formats[fmt]