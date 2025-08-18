from .dataset import Active_Event_Stereo_Dataset, DAVIS346_Stereo_Dataset

__datasets__ = {
    "eventstereo": Active_Event_Stereo_Dataset,
    "davis346stereo": DAVIS346_Stereo_Dataset,
}