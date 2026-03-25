# ── Training schedule ──────────────────────────────────────────────────────────
ITERATIONS          = 30_000
SH_DEGREE           = 3
LAMBDA_DSSIM        = 0.2
DENSIFY_FROM        = 500
DENSIFY_UNTIL       = 15_000
DENSIFY_INTERVAL    = 100
OPACITY_RESET_EVERY = 3_000
SH_UPSCALE_EVERY    = 1_000

# ── Adaptive density control ───────────────────────────────────────────────────
DENSIFY_GRAD_THRESHOLD = 0.0002
DENSIFY_PERCENT_DENSE  = 0.01
DENSIFY_MAX_SCREEN_SIZE = 20
PRUNE_MIN_OPACITY      = 0.005
OPACITY_RESET_TARGET   = 0.01

# ── Optimizer ─────────────────────────────────────────────────────────────────
ADAM_EPS             = 1e-15
INIT_OPACITY         = 0.1
LR_XYZ               = 0.00016
LR_SH_BAND0          = 0.0025
LR_SH_BANDS_REST     = 0.000125
LR_OPACITY           = 0.05
LR_SCALING           = 0.005
LR_ROTATION          = 0.001
