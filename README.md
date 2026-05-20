# merit-aih-xray

## Dataset
* Tuberculosis (per diferenciar del fet a classe?) 3500 normal + 700 tuberculosis: https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset
* Pneumonia (similar a classe, però només pneumonia/normal) 1300 normal + 3800 pneumonia: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Problem definition
_Si és tuberculosis:_

Binary classifier between tuberculosis (TB) or normal, with class imbalance (700 TB against 3500 normal images)

## Outline??
1. [x] Dataset analysis: Class distribution, image samples, dataset splits
2. [ ] **Això potser no fa falta? tampoc aporta res de nou** Baseline model: simple CNN or pretrained ResNet with no special things. **Serves as baseline, reference point**
3. [x] Transfer learning: with one/two models. Could allow for model comparison. Should consider same training conditions (split, optimizer, epochs, image size). Pick the best out of the two for the other steps.
    * [x] ResNet18
    * [x] EfficientNetB0
4. [x] Imbalance comparison: compare model performance with:
    * no correction (dataset as it is)
    * class weights
    * oversampling TB (duplicate images)
    * downsampling NORMAL (ignore normal images)
    * data augmentation for TB (add noise, rotations, zoom, brightness/contrast). Careful with flipping, as it changes biological structure (heart is not on the right side)
5. [ ] Hyperparameter tuning. Tune learning rate, batch size, and frozen / partially unfrozen backbone
    * F1 score for different param values? (lab 2 alguna cosa similar amb lasso)
6. [ ] Evaluation: 
    * conf. matrix, precision, recall, F1, AUC.
    * training / validation loss curves, accuracy curves,
    * RECALL és important per detecció de TB, no volem falsos negatius
7. [ ] Explainability:
    * GradCAM, LIME, SHAP ja s'han vist a classe. Es podrien utilitzar igualment
    * Occlusion sensitivity: no l'hem vist. Per PyTorch hi ha llibreria Captum
    * Important to focus on what the model is using: is it the lungs section, or instead it focuses on image borders, labels in the image, or other artifacts?
8. Error analysis???
    * Analyze FN, FP
    * data augmentation reduced FN
