import matplotlib.pyplot as plt
import numpy as np

line_color_list=["b","g","r","c","m","y","k","w"]
line_style_list=["-","--","-.",":"]
def model_evaluation_metrics_curves(fig,record_scores):

        ax=fig.add_subplot(111)
        ax.plot(record_scores['epoch'],record_scores["train_scores"],color='red',label='train')
        ax.plot(record_scores['epoch'], record_scores['val_scores'],color='skyblue',label='validation')
        ax.set_title("train and validation ")
        ax.legend()



def model_evaluation_metrics_curves_vanilla(fig,record_scores,score_metrics,subplot_numCols=2):
    subplot_nums=len(score_metrics)
    subplot_numRows = int(np.ceil(subplot_nums / subplot_numCols))

    subplot_iter=1
    for score_metric in score_metrics:
        ax=fig.add_subplot(subplot_numRows,subplot_numCols,subplot_iter)
        ax.plot(record_scores['epoch'],record_scores[score_metric]['train_scores'],color='red',label='train')
        ax.plot(record_scores['epoch'], record_scores[score_metric]['val_scores'],color='skyblue',label='validation')
        ax.set_title(score_metric)
        ax.legend()
        subplot_iter += 1