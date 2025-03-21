import torch.nn as nn
import timm

class SpeciesBehaviorCNN(nn.Module):
    def __init__(self, num_species_classes, num_behavior_classes=2):
        super(SpeciesBehaviorCNN, self).__init__()
        
        # Load Xception model from timm
        self.backbone = timm.create_model("xception", pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the final FC layer
        
        # Define classifiers
        self.species_classifier = nn.Linear(self.backbone.num_features, num_species_classes)
        self.behavior_classifier = nn.Linear(self.backbone.num_features, num_behavior_classes)  # Binary classifier for behavior
    
    def forward(self, x):
        features = self.backbone(x)
        species_out = self.species_classifier(features)
        behavior_out = self.behavior_classifier(features)
        return species_out, behavior_out