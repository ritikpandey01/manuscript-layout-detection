# Kaggle Training Steps for YOLOv8

Follow these instructions to train the YOLO model on Kaggle:

### 1. Create a Kaggle Dataset from `dataset/`
1. Compress the project's `dataset/` folder into a zip file:
   ```bash
   zip -r dataset.zip dataset/
   ```
2. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) and click **New Dataset**.
3. Upload `dataset.zip`, name it `manuscript-layout-dataset`, and click **Create**.

### 2. Create and Configure the Notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) and click **New Notebook**.
2. Click **File -> Import Notebook** and upload `method_yolo/kaggle_train.ipynb`.
3. In the right sidebar:
   - **Data**: Click **Add Input**, search for your `manuscript-layout-dataset` and add it.
   - **Accelerator**: Select **GPU T4 x2** (or GPU P100).
   - **Internet**: Toggle to **On** (required to install `ultralytics` and download pretrained weights).

### 3. Run Training
1. Click **Run All**.
2. Training runs for ~60 epochs with early stopping (`patience=20`) and takes ~15–25 minutes on a T4 GPU.
3. The notebook automatically evaluates `best.pt` on `val`, `test_synthetic`, and `real_test`, and packages all artifacts into `/kaggle/working/outputs.zip`.

### 4. Download and Install Weights
1. In the right panel under **Output** (`/kaggle/working`), locate `outputs.zip`.
2. Click the three dots next to `outputs.zip` and select **Download**.
3. Extract `best.pt` from the zip into your local repository at:
   ```
   method_yolo/weights/best.pt
   ```
4. Once `best.pt` is in place, you can run inference using the trained model:
   ```bash
   python inference.py --input ./dataset/real_test/images --output ./results --method yolo
   python inference.py --input ./dataset/real_test/images --output ./results --method hybrid
   ```
