import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from pandas import DataFrame
import seaborn as sns

def plot_line_or_scatter(type_plot, x_data, y_data, x_label="", y_label="", title="", color = "r", yscale_log=False):
    # Create the plot object
    _, ax = plt.subplots()
    
    if type_plot == 'scatter':
        ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)
    
    else:
        ax.plot(x_data, y_data, lw = 2, color = color, alpha = 1)
        
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label) 
    
    
def visualize_3d(X,y,algorithm="tsne",title="Data in 3D"):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    if algorithm=="tsne":
        reducer = TSNE(n_components=3,random_state=47,n_iter=400,angle=0.6)
    elif algorithm=="pca":
        reducer = PCA(n_components=3,random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")
    
    if X.shape[1]>3:
        X = reducer.fit_transform(X)
    else:
        if type(X)==DataFrame:
        	X=X.values
    
    marker_shapes = ["circle","diamond", "circle-open", "square",  "diamond-open", "cross","square-open",]
    traces = []
    for hue in np.unique(y):
        X1 = X[y==hue]

        trace = go.Scatter3d(
            x=X1[:,0],
            y=X1[:,1],
            z=X1[:,2],
            mode='markers',
            name = str(hue),
            marker=dict(
                size=12,
                symbol=marker_shapes.pop(),
                line=dict(
                    width=int(np.random.randint(3,10)/10)
                ),
                opacity=int(np.random.randint(6,10)/10)
            )
        )
        traces.append(trace)


    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='Dim 1'),
            yaxis=dict(
                title='Dim 2'),
            zaxis=dict(
                title='Dim 3'), ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    iplot(fig)
    
    
def visualize_2d(X,y,algorithm="tsne",title="Data in 2D",figsize=(8,8)):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    if algorithm=="tsne":
        reducer = TSNE(n_components=2,random_state=47,n_iter=400,angle=0.6)
    elif algorithm=="pca":
        reducer = PCA(n_components=2,random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")
    if X.shape[1]>2:
        X = reducer.fit_transform(X)
    else:
        if type(X)==DataFrame:
            X=X.values
    f, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=figsize)
    sns.scatterplot(X[:,0],X[:,1],hue=y,ax=ax1);
    ax1.set_title(title);
    plt.show();
    
# Классификация с шумом    
def Div_plate(clf, flip_level):
    from sklearn.datasets import make_classification # для генерации данных
    from sklearn.model_selection import train_test_split # для разделения на обучение и тест
    import numpy as np
    import seaborn as sns # для простого отображения
    import matplotlib.pyplot as plt #
    # создаем данные
    X,y = make_classification(n_samples=10000, # число примеров
                              n_features=2, # число признаков (атрибутов)
                              n_informative=2, # из них информативных
                              n_redundant=0, # из них не информативных
                              n_repeated=0, # из них повторяющихся
                              n_classes=2, # число классов
                              n_clusters_per_class=1, # число кластеров на класс
                              class_sep=2, # влияет на расстояние между кластерами
                              flip_y=flip_level, # доля ошибок (шума)
                              weights=[0.5,0.5], # пропорции числа данных в классах
                              random_state=17) # 

    # разделяем на обучающие и тестовые, случайно
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)
    # рисуем данные
    plt.subplots();#
    ax1=plt.gca();#
    sns.scatterplot(X_train[:,0],X_train[:,1],hue=y_train,ax=ax1);# обучающие
    sns.scatterplot(X_test[:,0],X_test[:,1],hue=y_test,ax=ax1,marker="+");# тестовые
    ax1.set_title("Data With Noise");#
    plt.show();#
    # Создаем классификатор на основе случайного леса. Изменяйте параметры и смотрите как это влияет на обучение
    clf.fit(X_train, y_train) # обучаем 
    y_pred = clf.predict(X_test) # проверяем на тесттовых данных
    score=clf.score(X_test, y_test) # считаем среднюю точность
    print('Score=',score)
    ind=y_test==y_pred; # индексы совпадений результата классификации и меток классов
    plt.subplots(); 
    ax2=plt.gca();
    # рисуем "правильно" распознанные примеры
    sns.scatterplot(X_test[ind,0],X_test[ind,1],hue=y_test[ind],ax=ax2);
    # рисуем "неправильно" распознанные примеры
    sns.scatterplot(X_test[~ind,0],X_test[~ind,1],hue=y_pred[~ind],ax=ax2,marker="+");
    #sns.scatterplot(X_test[:,0],X_test[:,1],hue=y_test[:],ax=ax2,marker="+");
    ax2.set_title("With Noise");
    # считаем и рисуем разделяющую поверхность.
    plot_step=0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1 # немного измененные минимальное и максимальые значения превого атрибута 
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1 # немного измененные минимальное и максимальые значения второго атрибута 
    # считаем прямоугольную сетку возможных значений этих атрибутов
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), #
                         np.arange(y_min, y_max, plot_step)) #
    # считаем выход классификатора для всех примеров сетки
    # не забыв что массивы данных нужно привести к требуемому размеру.
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) # и преобразуем обратно в исходному размеру
    # рисуем разделяющую поверхность
    cs = plt.contourf(xx, yy, Z, levels=1, colors=['blue','orange'],alpha=0.3) # рисуем контурную карту
    plt.show();
    


def byte_to_float(byte):
    # byte -> int16(PCM_16) -> float32
    return pcm2float(np.frombuffer(byte,dtype=np.int16), dtype='float32')

def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    if sig.dtype.kind not in 'iu':
        raise TypeError("'sig' must be an array of integers")
    dtype = np.dtype(dtype)
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max
