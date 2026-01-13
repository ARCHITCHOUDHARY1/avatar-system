import torch
from transformers import pipeline

class FreeVisionModels:
    
    def __init__(self):
        self.models = {}
    
    def load_mediapipe(self):
        import mediapipe as mp
        
        mp_face_detection = mp.solutions.face_detection
        mp_face_mesh = mp.solutions.face_mesh
        mp_pose = mp.solutions.pose
        
        return {
            'face_detection': mp_face_detection.FaceDetection(min_detection_confidence=0.5),
            'face_mesh': mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1),
            'pose': mp_pose.Pose(static_image_mode=True, model_complexity=1)
        }
    
    def load_blip(self):
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16
        )
        return processor, model
    
    def load_dinov2(self):
        from transformers import AutoImageProcessor, AutoModel
        
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained(
            'facebook/dinov2-base',
            torch_dtype=torch.float16
        )
        return processor, model
    
    def load_gfpgan(self):
        from basicsr.archs.gfpganv1_arch import GFPGANv1
        import torch
        
        model = GFPGANv1(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=1,
            resample_kernel=(1, 3, 3, 1),
            decoder_load_path=None,
            fix_decoder=True,
            num_mlp=8,
            lr_mlp=0.01,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True
        )
        
        # Load pretrained weights
        checkpoint = torch.load('models/gfpgan/GFPGANv1.3.pth', map_location='cpu')
        model.load_state_dict(checkpoint['params'])
        
        return model