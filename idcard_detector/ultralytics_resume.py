from ultralytics.models.yolo.pose import PoseTrainer

from .training import ResumeInPlaceTrainerMixin


class ResumeInPlacePoseTrainer(ResumeInPlaceTrainerMixin, PoseTrainer):
    """Pose trainer that resumes into the configured output directory."""
