def R2PLUS1D(VIDEOPATH):
    import torch
    import json
    from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
    from pytorchvideo.data.encoded_video import EncodedVideo

    # Load pretrained R2Plus1D_18 from torchvision
    weights = R2Plus1D_18_Weights.DEFAULT
    model = r2plus1d_18(weights=weights)
    model.eval()

    transform = weights.transforms()

    # Load video
    video = EncodedVideo.from_path(VIDEOPATH, decoder="decord")
    video_data = video.get_clip(start_sec=0, end_sec=16/30)
    clip = video_data["video"] 

    clip = clip.permute(1, 0, 2, 3)
    clip = transform(clip)
    clip = clip.unsqueeze(0)

    # Inference
    with torch.no_grad():
        out = model(clip)

    # Top-5 predictions
    top5 = torch.topk(out, 5).indices.squeeze().tolist()

    # Load class names
    with open("kinetics_classnames.json") as f:
        class_to_id = json.load(f)

    # Translate ID to label
    id_to_class = {str(v): k for k, v in class_to_id.items()}

    # print top 5 predictions
    label_names = [id_to_class[str(i)] for i in top5]
    return ", ".join(label_names)
