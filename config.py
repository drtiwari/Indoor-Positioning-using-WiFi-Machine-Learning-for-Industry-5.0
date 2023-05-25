# ============================================================================ #
# FILE LOCATION DETAILS
from os.path import join, dirname

from typing import Final

# DIRECTORY PATH
main_loc: Final[str] = dirname(__file__)

########################
# MAIN FOLDERS
dir_IDAT: Final[str] = "INP_DATA"
dir_RDAT: Final[str] = "RUN_DATA"
dir_FUN: Final[str] = "FUNCTIONS"
dir_ANA: Final[str] = "ANALYSIS"
dir_PRD: Final[str] = "PREDICTIONS"

#######################
# PRIMARY LOCATIONS
loc_idat: Final[str] = join(main_loc, dir_IDAT)
loc_rdat: Final[str] = join(main_loc, dir_RDAT)
loc_fun: Final[str] = join(main_loc, dir_FUN)
loc_ana: Final[str] = join(main_loc, dir_ANA)
loc_prd: Final[str] = join(main_loc, dir_PRD)

#######################
# PRIMARY LOCATIONS
targets_bf = ["BUILDINGID", "FLOOR"]
targets_ll = ["LONGITUDE", "LATITUDE"]
