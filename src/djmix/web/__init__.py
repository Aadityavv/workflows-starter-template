"""HTTP layer: a local web app over the same analysis, planning, and render code.

Nothing here re-implements audio logic. The web app is a client of the exact
modules the CLI uses, which is the point of keeping `audio/`, `planning/`, and
`render/` free of interface concerns.
"""
