import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Загрузка данных
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Предобработка данных
def preprocess_data(data, is_training=True, preprocessor=None):
    # Разделение на признаки и целевую переменную
    X = data.drop(['Patient ID', 'Diagnosis'], axis=1)
    y = data['Diagnosis'].map({'No Glaucoma': 0, 'Glaucoma': 1})
    
    # Определяем категориальные и числовые колонки
    categorical_columns = ['Gender', 'Family History', 'Medical History', 'Medication Usage', 
                         'Cataract Status', 'Angle Closure Status', 'Visual Symptoms', 'Glaucoma Type']
    numerical_columns = ['Age', 'Visual Acuity Measurements', 'Intraocular Pressure (IOP)', 
                        'Cup-to-Disc Ratio (CDR)', 'Visual Field Test Results', 
                        'Optical Coherence Tomography (OCT) Results', 'Pachymetry']
    
    # Создаем копию данных для преобразований
    X_processed = X.copy()
    
    # Обработка числовых признаков
    for col in numerical_columns:
        if X_processed[col].dtype == 'object':
            # Извлекаем числа из строк, заменяем нечисловые значения на NaN
            X_processed[col] = pd.to_numeric(X_processed[col].str.extract(r'(\d+\.?\d*)')[0], errors='coerce')
            # Заполняем пропуски медианным значением
            X_processed[col] = X_processed[col].fillna(X_processed[col].median())
        else:
            X_processed[col] = X_processed[col].astype(float)
    
    # Обработка выбросов в числовых признаках
    for col in numerical_columns:
        Q1 = X_processed[col].quantile(0.25)
        Q3 = X_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_processed[col] = X_processed[col].clip(lower_bound, upper_bound)
    
    # Создаем новые признаки на основе медицинских знаний
    # Возрастные группы
    age_bins = [0, 40, 50, 60, 70, 100]
    age_labels = ['0-40', '40-50', '50-60', '60-70', '70+']
    X_processed['Age_Group'] = pd.cut(X_processed['Age'], bins=age_bins, labels=age_labels)
    
    # ВГД группы
    iop_bins = [0, 21, 25, 30, 100]
    iop_labels = ['Normal', 'Borderline', 'High', 'Very High']
    X_processed['IOP_Group'] = pd.cut(X_processed['Intraocular Pressure (IOP)'],
                                    bins=iop_bins, labels=iop_labels)
    
    # CDR группы
    cdr_bins = [0, 0.5, 0.6, 0.7, 1.0]
    cdr_labels = ['Normal', 'Borderline', 'High', 'Very High']
    X_processed['CDR_Group'] = pd.cut(X_processed['Cup-to-Disc Ratio (CDR)'],
                                    bins=cdr_bins, labels=cdr_labels)
    
    # Взаимодействие признаков
    X_processed['Age_IOP'] = X_processed['Age'] * X_processed['Intraocular Pressure (IOP)']
    X_processed['Age_CDR'] = X_processed['Age'] * X_processed['Cup-to-Disc Ratio (CDR)']
    X_processed['IOP_CDR'] = X_processed['Intraocular Pressure (IOP)'] * X_processed['Cup-to-Disc Ratio (CDR)']
    
    # Полиномиальные признаки
    X_processed['Age_squared'] = X_processed['Age'] ** 2
    X_processed['IOP_squared'] = X_processed['Intraocular Pressure (IOP)'] ** 2
    
    # Обновляем списки признаков
    numerical_columns = [col for col in X_processed.columns 
                        if col not in categorical_columns 
                        and col not in ['Age_Group', 'IOP_Group', 'CDR_Group']]
    
    categorical_columns = categorical_columns + ['Age_Group', 'IOP_Group', 'CDR_Group']
    
    # Создаем или используем существующий препроцессор
    if is_training:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
            ])
        X_processed = preprocessor.fit_transform(X_processed)
    else:
        X_processed = preprocessor.transform(X_processed)
    
    return X_processed, y, preprocessor

# Создание модели
def create_model(input_dim):
    model = Sequential()
    
    # Входной слой
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    
    # Первый блок
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Второй блок
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Выходной слой
    model.add(Dense(1, activation='sigmoid'))
    
    # Компиляция с оптимизированными параметрами
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), 
                tf.keras.metrics.Precision(), 
                tf.keras.metrics.Recall(),
                tf.keras.metrics.F1Score()]
    )
    
    return model

# Обучение модели
def train_model(model, X_train, y_train, X_test, y_test):
    # Вычисляем веса классов
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))
    
    # Применяем SMOTE для балансировки классов
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # Ранняя остановка
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Уменьшение learning rate
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Сохранение лучшей модели
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    # Обучение модели
    history = model.fit(
        X_res, y_res,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        class_weight=class_weights,
        verbose=1
    )
    
    return history

# Визуализация результатов
def plot_results(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Точность на обучении')
    plt.plot(history.history['val_accuracy'], label='Точность на валидации')
    plt.title('Точность модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Потери на обучении')
    plt.plot(history.history['val_loss'], label='Потери на валидации')
    plt.title('Потери модели')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

def cross_validate_model(data, n_splits=5):
    """Выполнение кросс-валидации модели"""
    print("\nНачинаем кросс-валидацию...")
    
    # Инициализируем KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Списки для хранения метрик
    fold_accuracies = []
    fold_losses = []
    fold_precisions = []
    fold_recalls = []
    fold_f1_scores = []
    
    # Выполняем кросс-валидацию
    for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
        print(f"\nФолд {fold}/{n_splits}")
        
        # Разделяем данные
        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        # Предобработка данных для текущего фолда
        X_train, y_train, preprocessor = preprocess_data(train_data, is_training=True)
        X_val, y_val, _ = preprocess_data(val_data, is_training=False, preprocessor=preprocessor)
        
        # Создаем и обучаем модель
        model = create_model(X_train.shape[1])
        history = train_model(model, X_train, y_train, X_val, y_val)
        
        # Оцениваем модель
        val_loss, val_acc, val_auc, val_precision, val_recall, val_f1 = model.evaluate(X_val, y_val, verbose=0)
        
        # Сохраняем метрики
        fold_accuracies.append(val_acc)
        fold_losses.append(val_loss)
        fold_precisions.append(val_precision)
        fold_recalls.append(val_recall)
        fold_f1_scores.append(val_f1)
        
        print(f"Фолд {fold} - Точность: {val_acc:.4f}, Потери: {val_loss:.4f}")
        print(f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
    
    # Выводим средние результаты
    print("\nРезультаты кросс-валидации:")
    print(f"Средняя точность: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
    print(f"Средние потери: {np.mean(fold_losses):.4f} ± {np.std(fold_losses):.4f}")
    print(f"Средний Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
    print(f"Средний Recall: {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
    print(f"Средний F1-score: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")
    
    # Визуализируем результаты кросс-валидации
    plot_cross_validation_results(fold_accuracies, fold_losses, fold_precisions, fold_recalls, fold_f1_scores, n_splits)
    
    return np.mean(fold_accuracies), np.std(fold_accuracies)

def plot_cross_validation_results(accuracies, losses, precisions, recalls, f1_scores, n_splits):
    plt.figure(figsize=(15, 12))
    
    # График точности по фолдам
    plt.subplot(3, 2, 1)
    plt.bar(range(1, n_splits + 1), accuracies)
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--', label='Средняя точность')
    plt.title('Точность по фолдам')
    plt.xlabel('Фолд')
    plt.ylabel('Точность')
    plt.legend()
    
    # График потерь по фолдам
    plt.subplot(3, 2, 2)
    plt.bar(range(1, n_splits + 1), losses)
    plt.axhline(y=np.mean(losses), color='r', linestyle='--', label='Средние потери')
    plt.title('Потери по фолдам')
    plt.xlabel('Фолд')
    plt.ylabel('Потери')
    plt.legend()
    
    # График precision по фолдам
    plt.subplot(3, 2, 3)
    plt.bar(range(1, n_splits + 1), precisions)
    plt.axhline(y=np.mean(precisions), color='r', linestyle='--', label='Средний precision')
    plt.title('Precision по фолдам')
    plt.xlabel('Фолд')
    plt.ylabel('Precision')
    plt.legend()
    
    # График recall по фолдам
    plt.subplot(3, 2, 4)
    plt.bar(range(1, n_splits + 1), recalls)
    plt.axhline(y=np.mean(recalls), color='r', linestyle='--', label='Средний recall')
    plt.title('Recall по фолдам')
    plt.xlabel('Фолд')
    plt.ylabel('Recall')
    plt.legend()
    
    # График F1-score по фолдам
    plt.subplot(3, 2, 5)
    plt.bar(range(1, n_splits + 1), f1_scores)
    plt.axhline(y=np.mean(f1_scores), color='r', linestyle='--', label='Средний F1-score')
    plt.title('F1-score по фолдам')
    plt.xlabel('Фолд')
    plt.ylabel('F1-score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_importance(model, X_train, feature_names):
    """Анализ важности признаков с помощью SHAP"""
    explainer = shap.DeepExplainer(model, X_train)
    shap_values = explainer.shap_values(X_train)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, feature_names=feature_names)
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_data(data):
    """Анализ данных перед обучением"""
    print("\n=== Анализ данных ===")
    
    # Анализ баланса классов
    print("\nРаспределение классов:")
    class_dist = data['Diagnosis'].value_counts(normalize=True)
    print(class_dist)
    
    # Анализ числовых признаков
    numerical_columns = ['Age', 'Visual Acuity Measurements', 'Intraocular Pressure (IOP)', 
                        'Cup-to-Disc Ratio (CDR)', 'Visual Field Test Results', 
                        'Optical Coherence Tomography (OCT) Results', 'Pachymetry']
    
    # Создаем копию данных для анализа
    data_analysis = data.copy()
    
    # Преобразуем числовые признаки
    for col in numerical_columns:
        if data_analysis[col].dtype == 'object':
            # Извлекаем числа из строк
            data_analysis[col] = data_analysis[col].str.extract(r'(\d+\.?\d*)').astype(float)
    
    # Преобразуем целевую переменную в числовой формат
    data_analysis['Diagnosis'] = data_analysis['Diagnosis'].map({'No Glaucoma': 0, 'Glaucoma': 1})
    
    print("\nСтатистика числовых признаков:")
    print(data_analysis[numerical_columns].describe())
    
    # Корреляция с целевой переменной
    print("\nКорреляция с целевой переменной:")
    correlations = data_analysis[numerical_columns + ['Diagnosis']].corr()['Diagnosis'].sort_values(ascending=False)
    print(correlations)
    
    # Визуализация корреляций
    plt.figure(figsize=(12, 8))
    sns.heatmap(data_analysis[numerical_columns + ['Diagnosis']].corr(), annot=True, cmap='coolwarm')
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    
    # Анализ категориальных признаков
    categorical_columns = ['Gender', 'Family History', 'Medical History', 'Medication Usage', 
                         'Cataract Status', 'Angle Closure Status', 'Visual Symptoms', 'Glaucoma Type']
    
    print("\nРаспределение категориальных признаков:")
    for col in categorical_columns:
        print(f"\n{col}:")
        print(data[col].value_counts(normalize=True))

def main():
    # Загрузка данных
    data = load_data('glaucoma_dataset.csv')
    
    # Анализ данных
    analyze_data(data)
    
    # Предобработка данных
    X_train, y_train, preprocessor = preprocess_data(data, is_training=True)
    X_test, y_test, _ = preprocess_data(data, is_training=False, preprocessor=preprocessor)
    
    # Создание и обучение модели
    model = create_model(X_train.shape[1])
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Визуализация результатов
    plot_results(history)
    
    # Оценка модели на тестовых данных
    test_loss, test_accuracy, test_auc, test_precision, test_recall, test_f1 = model.evaluate(X_test, y_test)
    print(f'\nРезультаты на тестовой выборке:')
    print(f'Потери: {test_loss:.4f}')
    print(f'Точность: {test_accuracy:.4f}')
    print(f'AUC: {test_auc:.4f}')
    print(f'Precision: {test_precision:.4f}')
    print(f'Recall: {test_recall:.4f}')
    print(f'F1-score: {test_f1:.4f}')
    
    # Полный отчет о классификации
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print('\nПолный отчет о классификации:')
    print(classification_report(y_test, y_pred))
    
    # Анализ важности признаков
    feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    analyze_feature_importance(model, X_train, feature_names)
    
    # Кросс-валидация
    cross_validate_model(data)

if __name__ == "__main__":
    main() 