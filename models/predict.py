from albumentations.pytorch import ToTensorV2
import torchvision
import tempfile
import jpegio
from PIL import Image
import matplotlib.pyplot as plt
from dtd import seg_dtd
from swins import *
from torch.autograd import Variable

totsr = ToTensorV2()
toctsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

def display(image, mask, predicted=False):
    if predicted:
        title = ["Predicted Mask", "Blended Image with Predicted Mask Overlay"]
    else:
        title = ["Mask", "Blended Image with Mask Overlay"]

    # Create a red overlay for tampered areas
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask == 255] = [255, 0, 0, 128]

    #Convert the overlay to an image
    overlay_image = Image.fromarray(overlay)

    # Blend the overlay with the original image
    blended_image = Image.blend(image, overlay_image, alpha=0.3)

    # Display images: original, predicted mask, and blended overlay
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title(title[0])
    ax[1].axis('off')

    ax[2].imshow(blended_image)
    ax[2].set_title(title[1])
    ax[2].axis('off')

    plt.show()


def visualize_dct_coef(image, dct_coef, apply_log=True):
    """
    Displays the original image, DCT Coefficients, and the overlay of DCT coefficients on the original image.

    Parameters:
    - image: PIL Image object or np.ndarray: Original image
    - dct_coef: torch.Tensor or np.ndarray: DCT Coefficient matrix of shape (1, 512, 512)
    - apply_log: bool: Apply log transformation to DCT Coefficients for visualizations
    """
    # Convert dct_coef to numpy array if it's a tensor
    if isinstance(dct_coef, torch.Tensor):
        dct_coef = dct_coef.squeeze(0).cpu().numpy()

    # Apply log transformation to DCT Coefficients
    dct_display = np.log1p(np.abs(dct_coef)) if apply_log else np.abs(dct_coef)

    # Prepare the overlay with blue color and transparency
    overlay = np.zeros((dct_display.shape[0], dct_display.shape[1], 4), dtype=np.uint8)
    overlay[:, :, 2] = (dct_display / dct_display.max() * 255).astype(np.uint8)   # Blue
    overlay[:, :, 3] = 128  # Set alpha for transparancy

    # Convert the overlay to an image
    overlay_image = Image.fromarray(overlay, mode='RGBA')
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Blend original imgae with overlay
    blended_image = Image.blend(image.convert("RGBA"), overlay_image, alpha=0.6)

    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Display original image
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis('off')

    # Display DCT Coefficients
    dct_img = ax[1].imshow(dct_display, cmap='viridis')
    ax[1].set_title("DCT Coefficients" + (" (Log Scale)" if apply_log else ""))
    ax[1].axis('off')
    fig.colorbar(dct_img, ax=ax[1], orientation='horizontal', fraction=0.046, pad=0.04)

    # Display overlay
    ax[2].imshow(blended_image)
    ax[2].set_title("DCT Coefficients Overlay on Original Image")
    ax[2].axis('off')

    plt.tight_layout()
    plt.show()

def predict_from_file(filepath, model, visualize_dct=False):
    if filepath.lower().endswith('jpg') or filepath.lower().endswith('jpeg'):
        jpg = jpegio.read(filepath)
    else:
        # For non jpeg files, convert and save as high-quality JPEG temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img = Image.open(filepath).convert('L')
            img.save(tmp.name, "JPEG", quality=100, subsampling=0)
            jpg = jpegio.read(tmp.name)
    
    dct = jpg.coef_arrays[0].copy()
    qtb = torch.LongTensor(jpg.quant_tables[0])
  
  # Load RGB data and prepare for model
    image = Image.open(filepath).convert('RGB')
    data = toctsr(image)
    dct_coef = torch.from_numpy(np.clip(np.abs(dct), 0, 20))
    if visualize_dct:
        visualize_dct_coef(image, dct_coef)

    device = 'cuda'
    data, dct_coef, qtb = Variable(data.unsqueeze(0).to(device)), Variable(dct_coef.unsqueeze(0).to(device)), Variable(qtb.unsqueeze(0).unsqueeze(0).to(device))

    with torch.no_grad():
        pred = model(data, dct_coef, qtb)
        predt = pred.argmax(1)

    mask = predt.squeeze(0).cpu().numpy() * 255
    return mask


if __name__ == '__main__':
    # Load the model
    model = seg_dtd('', 2).cuda()
    model = torch.nn.DataParallel(model)
    ckpt = torch.load('weights/dtd_doctamper.pth')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    image_file = 'test.jpg'

    pred_mask = predict_from_file(image_file, model, visualize_dct=True)
    display(Image.open(image_file).convert("RGBA"), pred_mask, predicted=True)