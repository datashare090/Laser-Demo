
import streamlit as st
from PIL import Image
import torch
import cv2
import numpy as np
from model.unet_model import UNet
from torchvision import datasets, transforms
from model.ResNet import resnet18
import os



st.title('Laser Seam Detection Demo')
STYLE = """
<style>
img {
    max-width: 100%;
}
</style>
"""
import toml
primaryColor = "#000000"
s = f"""
<style>
div.stButton > button:first-child {{ border: 5px solid {primaryColor}; background-color: #D0F0C0; }}
<style>
"""
st.markdown(s, unsafe_allow_html=True)

def main():

    st.sidebar.title("What to do")
    app_mode = st.sidebar.radio("Choose the app mode",
                                    ["Project Overview", "Classification", "Edge Detection"])

    if app_mode == "Project Overview":
        edge_img = Image.open('./edge.jpg')
        full_image = Image.open('./complete.jpg')
        header1=st.header("Importance")
        placeholder1 = st.image(edge_img, width=900, caption="Edge Location: Pixel Level Precision is required")

        header2=st.header("Project Overview")
        placeholder2=st.image(full_image, width=900, caption="(a) Classification module for identification of normal or "
                                                "abnormal edge and welded or non-welded edge.  (b) Edge detection "
                                                "module which is a modified UNET architecture. 3×3 convolution in the "
                                                "second last layer is replaced with 2×2 max pooling layer to get 1D output. "
                                                "The final output of (b) is 1×W, i.e., 1×43 where the boundary between0s"
                                                " and 1s indicates an edge.")
        header3 = st.header("Labeling Demo")

        video_file = open('./video.mp4', 'rb')
        video_bytes = video_file.read()
        placeholder3=st.video(video_bytes)


    elif app_mode == "Edge Detection":

        st.subheader('Welcome to Edge Detection Module')
        edge_detection()

    elif app_mode == "Classification":

        st.subheader('Welcome to Classification Module')
        classification()



def edge_detection():
    st.sidebar.markdown("# Data")
    data_type = st.sidebar.radio("Do you want to upload your own data or select images from given database?",
                                       ["Own Data", "Database"])
    if data_type=="Own Data":
        file_selector_O()
    if data_type=="Database":
        file_selector_DB()

def classification():

    st.sidebar.markdown("# Classification Type")
    data_type = st.sidebar.radio("Do you want to classify welded/non-welded image dataset or easy/difficult images?",
                                       ["Welded/Non-Welded", "Easy/Difficult"])

    if data_type=="Welded/Non-Welded":

        welded_nonwelded()
    if data_type=="Easy/Difficult":

        easy_difficult()

def easy_difficult():
    easy_difficult_DB()

def easy_difficult_DB():
    folder_path = "./Datasets/easy_difficult"
    filenames = os.listdir(folder_path)
    filenames1 = ['None']
    list = filenames1 + filenames

    selected_filename = st.sidebar.selectbox('Select a file', list)

    if selected_filename != 'None':
        file_path = os.path.join(folder_path, selected_filename)
        pil_img = Image.open(file_path)

        img_nd = preprocess(pil_img)
        st.image(img_nd, caption="Full Image")

        transform = transforms.Compose([
            transforms.CenterCrop(87),
            transforms.ToTensor()
        ])
        pil_img = Image.fromarray(np.uint8(img_nd)).convert('RGB')
        data = transform(pil_img)
        imgs = torch.unsqueeze(data, 0)
        pred = classification_evaluation('./checkpoint/classification/nonweld_weld/epoch290_model.pth.tar', imgs)
        ROI = img_nd[140:227, 287:330]
        col1, col2, col3 = st.columns(3)
        col2.image(ROI, width=150, caption="ROI")
        col1, col2, col3 = st.columns(3)
        classify = col2.button('Classify')
        if classify:


            if pred == 0:
                title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Edge is difficult to locate</p>'

                st.markdown(title, unsafe_allow_html=True)
            else:
                title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Edge is easy to locate</p>'

                st.markdown(title, unsafe_allow_html=True)


def easy_difficult_O():

    st.info("Try as many times as you want!")
    fileTypes = ["jpeg", "png", "jpg", "bmp"]
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader("Upload image", type=fileTypes)

    show_file = st.empty()
    if not file:
        show_file.info("Please upload an image of type: " + ", ".join(["jpeg", "png", "jpg", "bmp"]))
        return
    # if isinstance(file, BytesIO):
    #   show_file.image(file, caption="Full Image")

    pil_img = Image.open(file)

    img_nd = preprocess(pil_img)
    st.image(img_nd, caption="Full Image")
    transform = transforms.Compose([
        transforms.CenterCrop(87),
        transforms.ToTensor()
    ])
    pil_img = Image.fromarray(np.uint8(img_nd)).convert('RGB')
    data = transform(pil_img)
    imgs = torch.unsqueeze(data, 0)
    pred = classification_evaluation('./checkpoint/classification/difficult_easy/epoch500_model.pth.tar', imgs)
    ROI = img_nd[140:227, 287:330]
    col1, col2, col3 = st.columns(3)
    col2.image(ROI, width=150, caption="ROI")
    classify = col2.button('Classify')
    if classify:

        if pred == 0:
            title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Edge is difficult to locate</p>'

            st.markdown(title, unsafe_allow_html=True)
        else:
            title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Edge is easy to locate</p>'

            st.markdown(title, unsafe_allow_html=True)

        file.close()
        st.info("To preceed further please clear the data by pressing cross under the browse button")




def welded_nonwelded():
    welded_nonwelded_DB()

def welded_nonwelded_DB():

    folder_path = "./Datasets/welded_nonwelded"
    filenames = os.listdir(folder_path)
    filenames1 = ['None']
    list = filenames1 + filenames

    selected_filename = st.sidebar.selectbox('Select a file', list)

    if selected_filename != 'None':
        file_path = os.path.join(folder_path, selected_filename)
        pil_img = Image.open(file_path)

        img_nd = preprocess(pil_img)
        st.image(img_nd, caption="Full Image")

        transform = transforms.Compose([
            transforms.CenterCrop(87),
            transforms.ToTensor()
        ])
        pil_img = Image.fromarray(np.uint8(img_nd)).convert('RGB')
        data = transform(pil_img)
        imgs = torch.unsqueeze(data, 0)
        pred = classification_evaluation('./checkpoint/classification/nonweld_weld/epoch290_model.pth.tar', imgs)
        ROI = img_nd[140:227, 287:330]
        col1, col2, col3 = st.columns(3)
        col2.image(ROI, width=150, caption="ROI")
        classify = col2.button('Classify')
        if classify:
            if pred == 0:
                title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Its a non-welded image</p>'

                st.markdown(title, unsafe_allow_html=True)
            else:
                title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Its a welded image. No need to detect an edge</p>'

                st.markdown(title, unsafe_allow_html=True)

def welded_nonwelded_O():

    st.info("Try as many times as you want!")
    fileTypes = ["jpeg", "png", "jpg", "bmp"]
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader("Upload image", type=fileTypes)

    show_file = st.empty()
    if not file:
        show_file.info("Please upload an image of type: " + ", ".join(["jpeg", "png", "jpg", "bmp"]))
        return
    #if isinstance(file, BytesIO):
     #   show_file.image(file, caption="Full Image")

    pil_img = Image.open(file)

    img_nd = preprocess(pil_img)
    st.image(img_nd, caption="Full Image")
    transform = transforms.Compose([
        transforms.CenterCrop(87),
        transforms.ToTensor()
    ])
    pil_img = Image.fromarray(np.uint8(img_nd)).convert('RGB')
    data = transform(pil_img)
    imgs = torch.unsqueeze(data, 0)
    pred = classification_evaluation('./checkpoint/classification/nonweld_weld/epoch290_model.pth.tar', imgs)
    ROI = img_nd[140:227, 287:330]
    col1, col2, col3 = st.columns(3)
    col2.image(ROI, width=150, caption="ROI")

    classify = col2.button('Classify')
    if classify:


        if pred == 0:
            title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Its a non-welded image</p>'

            st.markdown(title, unsafe_allow_html=True)
        else:
            title = '<p style="font-family:Courier; text-align: center; color:Black; font-size: 20px;">Its a welded image. No need to detect an edge</p>'

            st.markdown(title, unsafe_allow_html=True)

        file.close()
        st.info("To preceed further please clear the data by pressing cross under the browse button")


def classification_evaluation(checkpoint_path,data):

    model = resnet18(in_channels=3, num_classes=2)
    #model.cuda(0)
    model.eval()
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu') )

    model.load_state_dict(checkpoint['state_dict'])
    #data = data.cuda(0)
    output = model(data)

    pred = output.data.max(1, keepdim=True)[1]

    return pred.item()



def preprocess(pil_img):
    img_nd = np.array(pil_img)

    if len(img_nd.shape)<3:
        img_nd=cv2.cvtColor(img_nd, cv2.COLOR_GRAY2RGB)
    # img_nd=pil_img
    if len(img_nd.shape) == 2:
        img_nd = np.expand_dims(img_nd, axis=2)
    return img_nd


def HWCtoCHM(img_nd):
    # HWC to CHW
    img_trans = img_nd.transpose((2, 0, 1))
    if img_trans.max() > 1:
        img_trans = img_trans / 255

    return img_trans

def file_selector_O():


    st.info("Try as many times as you want!")
    fileTypes = ["jpeg", "png", "jpg","bmp"]
    st.markdown(STYLE, unsafe_allow_html=True)
    file = st.file_uploader("Upload image", type=fileTypes)
    show_file = st.empty()
    if not file:
        show_file.info("Please upload an image of type: " + ", ".join(["jpeg", "png", "jpg","bmp"]))
        return
    #if isinstance(file, BytesIO):
     #   col1.show_file.image(file,caption="Full Image")

    pil_img = Image.open(file)
    img_nd=preprocess(pil_img)

    st.image(img_nd,caption="Full Image")
    ROI = img_nd[140:227, 287:330]

    ROI_CHW = HWCtoCHM(ROI)
    col1, col2, col3 = st.columns(3)
    col2.image(ROI, width=200, caption="ROI")
    edge = col2.button("Detect Edge")

    if edge:


        edge_pixel=find_edge(ROI_CHW)

        image=img_nd.copy()
        cv2.line(image, (287 + edge_pixel, 80), (287 + edge_pixel, 300), (255,255,0), 1)
        cv2.putText(image, 'Prediction', (450, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(255,255,0), thickness=2)
        cv2.rectangle(image, (287, 227), (330, 140), (0, 0, 255), 1)

        st.image(image, caption="Full Image with Edge")
        ROI_plot = ROI.copy()
        cv2.line(ROI_plot, (edge_pixel, 0), (edge_pixel, 87), (255, 0, 255), 1)

        col1, col2, col3 = st.columns(3)

        col2.image(ROI_plot, width=200, caption="ROI with Edge")

        file.close()

        st.info("To preceed further please clear the data by pressing cross under the browse button")


def file_selector_DB():


    folder_path = "./Datasets/Edge_Detection"
    filenames = os.listdir(folder_path)
    filenames1 = ['None']
    list=filenames1+filenames

    selected_filename = st.sidebar.selectbox('Select a file', list)
    if selected_filename!='None':
        file_path = os.path.join(folder_path, selected_filename)
        img_name = selected_filename
        gt_edge = int(img_name[10:12])
        pil_img = Image.open(file_path)
        img_nd = preprocess(pil_img)
        ROI = img_nd[140:227, 287:330]

        ROI_CHW = HWCtoCHM(ROI)
        st.image(img_nd, caption="Full Image")
        col1, col2, col3 = st.columns(3)
        col2.image(ROI, width=170, caption="ROI")
        edge= col2.button("Detect Edge")

        if edge:



            edge_pixel=find_edge(ROI_CHW)

            image=img_nd.copy()
            cv2.rectangle(image, (287, 227), (330, 140), (255, 0, 0), 1)
            cv2.line(image, (287 + gt_edge, 80), (287 + gt_edge, 300), (255, 0, 255), 1)
            cv2.putText(image, 'GT', (450, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 255), thickness=2)

            cv2.line(image, (287 + edge_pixel, 80), (287 + edge_pixel, 300), (255,255,0), 1)
            cv2.putText(image, 'Prediction', (450, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255,255,0), thickness=2)


            st.image(image,caption="Full Image with Edge")


            ROI_plot=ROI.copy()
            cv2.line(ROI_plot, (gt_edge, 0), (gt_edge, 87), (255, 0, 255), 1)
            cv2.line(ROI_plot, (edge_pixel, 0), (edge_pixel, 87), (255,255,0), 1)



            col1, col2, col3 = st.columns(3)

            col2.image(ROI_plot, width=170, caption="ROI with Edge")
            st.markdown(
                "<h1 style='text-align: center; color:Black; font-size: 20px;'> %d px difference b/w GT and Network</h1>" % (
                    abs(gt_edge - edge_pixel)), unsafe_allow_html=True)




def find_edge(test_img):

    model=UNet(n_channels=3, n_classes=1, bilinear=True)
    #torch.cuda.set_device(0)
    #model = model.cuda(0)

    checkpoint = torch.load('./checkpoint/detection/model_best.pth.tar',map_location=torch.device('cpu') )
    model.load_state_dict(checkpoint['state_dict'])
    dataset_test= torch.from_numpy(test_img).type(torch.FloatTensor)
    edge_pixel=validate(dataset_test, model)
    return edge_pixel


def validate(dataset_test, model):


    # switch to evaluate mode
    model.eval()


    with torch.no_grad():


        imgs = torch.unsqueeze(dataset_test,0)
        #imgs = imgs.cuda(0, non_blocking=True)
        # compute output
        with torch.no_grad():
            mask_pred = model(imgs)

        pred = torch.sigmoid(mask_pred)
        pred = (pred > 0.5).float()
        probs=pred
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )

        for i in range(len(probs)):

            probs=pred[i]
            probs = tf(probs.cpu())
            full_mask = probs.cpu().numpy()
            full_mask_ones = np.where(full_mask[0] == 1)
            full_mask_ones = full_mask_ones[1]
            edge_pixel = full_mask_ones[0]

    return edge_pixel



if __name__ == "__main__":

    main()
