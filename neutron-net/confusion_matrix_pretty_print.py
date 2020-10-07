import numpy   as np
import seaborn as sn

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from   matplotlib.collections import QuadMesh

class ConfusionMatrixPrinter:
    """The ConfusionMatrixPrinter contains code for create confusion matrix plots"""
    
    @staticmethod
    def __get_new_fig(figure_num, fig_size=(9,9)):
        """Initialises graphics for confusion matrix plot.

        Args:
            figure_num (int): identifier for the figure.
            fig_size (tuple): size of the figure.

        Returns:
            matplotlib figure and axis objects.

        """
        fig = plt.figure(figure_num, fig_size, dpi=600)
        ax  = fig.gca() #Get the current axis.
        ax.cla() #Clear any existing plot.
        return fig, ax

    @staticmethod
    def __configcell_text_and_colors(array_df, row, column, text, face_colors, position, font_size, fmt, show_null_values=0):
        """Configures cell text and colours for the confusion matrix plot.

        Args:
            array_df (DataFrame): Pandas dataframe object to configure.
            row (int): index of row to modify.
            column (int): index of column to modify.
            text (Axes.text): text to delete.
            face_colors (object): quadmesh face colour object.
            position (int): position from left to right, bottom to top.
            font_size (float): size of the font for the text to add.
            fmt (type): format of the text to add.
            show_null_values (int): controls how null values are handled in the plot.

        Returns:
            Text elements to add and to delete.

        """
        text_add, text_del  = [], []
        cell_val = array_df[row][column] #Cell value
        ccl      = len(array_df[:,column]) #Current column length

        #last line and/or last column
        if (column == (ccl - 1)) or (row == (ccl - 1)):
            #Calculate totals and percentages
            if (cell_val != 0):
                if(column == ccl - 1) and (row == ccl - 1):
                    tot_rig = 0
                    for i in range(array_df.shape[0] - 1):
                        tot_rig += array_df[i][i]
                    per_ok = (float(tot_rig) / cell_val) * 100
                    
                elif (column == ccl - 1):
                    tot_rig = array_df[row][row]
                    per_ok  = (float(tot_rig) / cell_val) * 100
                    
                elif (row == ccl - 1):
                    tot_rig = array_df[column][column]
                    per_ok  = (float(tot_rig) / cell_val) * 100
                    
                per_err = 100 - per_ok
                
            else:
                per_ok = per_err = 0

            per_ok_s = ['%.2f%%'%(per_ok), '100%'][per_ok == 100]
            text_del.append(text) #Text to delete

            #Text to add
            font_prop   = fm.FontProperties(weight='bold', size=font_size)
            text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
            lis_txt     = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
            lis_kwa     = [text_kwargs]
            
            dic = text_kwargs.copy()
            dic['color'] = 'g'
            lis_kwa.append(dic)
            
            dic = text_kwargs.copy()
            dic['color'] = 'r'
            lis_kwa.append(dic)
            
            lis_pos = [(text._x, text._y-0.3), (text._x, text._y), (text._x, text._y+0.3)]
            for i in range(len(lis_txt)):
                newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
                text_add.append(newText)

            #Set background colour for sum cells (last row and last column)
            carr = [0.27, 0.30, 0.27, 1.0]
            if (column == ccl - 1) and (row == ccl - 1):
                carr = [0.17, 0.20, 0.17, 1.0]
            face_colors[position] = carr

        else:
            per = (float(cell_val) / array_df[-1][-1]) * 100
            if (per > 0):
                txt = '%s\n%.2f%%' %(cell_val, per)
            else:
                if (show_null_values == 0):
                    txt = ''
                elif (show_null_values == 1):
                    txt = '0'
                else:
                    txt = '0\n0.0%'
            text.set_text(txt)

            #Main diagonal
            if (column == row):
                #Set colour of the text in the diagonal to white.
                text.set_color('w')
                #Set background colour in the diagonal to blue.
                face_colors[position] = [0.35, 0.8, 0.55, 1.0]
            else:
                text.set_color('r')

        return text_add, text_del

    @staticmethod
    def __insert_totals(cm_dataframe):
        """Insert total column and total row (the last ones).

        Args:
            cm_dataframe (DataFrame): confusion matrix dataframe object to insert totals into.
        """
        sum_column = []
        for column in cm_dataframe.columns:
            sum_column.append(cm_dataframe[column].sum())

        sum_row = []
        for item_row in cm_dataframe.iterrows():
            sum_row.append(item_row[1].sum())

        cm_dataframe['sum_row'] = sum_row
        sum_column.append(np.sum(sum_row))
        cm_dataframe.loc['sum_column'] = sum_column

    @staticmethod
    def pretty_plot(cm_dataframe, annotate=True, cmap="Oranges", fmt='.2f', font_size=11,
          line_width=0.5, cbar=False, fig_size=(8,8), show_null_values=0, pred_val_axis='y'):
        """Prints confusion matrix with default layout.

        Args:
            cm_dataframe (DataFrame): Pandas dataframe without totals.
            annotate (Boolean): whether to print text in each cell.
            cmap (string): Oranges, Oranges_r, YlGnBu, Blues, RdBu,etc
            fmt (string): format of font text.
            font_size (float): size of font text.
            line_width (float): line width.
            cbar (Boolean): cbar parameter for seaborn heatmap.
            fig_size (tuple): size of the confusion matrix figure.
            show_null_values (int): controls how null values are handled in the plot.
            pred_val_axis (string): where to show the prediction values.

        """
        if (pred_val_axis in ('column', 'x')):
            xlbl = 'Predicted'
            ylbl = 'Actual'
        else:
            xlbl = 'Actual'
            ylbl = 'Predicted'
            cm_dataframe = cm_dataframe.T

        ConfusionMatrixPrinter.__insert_totals(cm_dataframe) #Create "Total" column
        fig, ax1 = ConfusionMatrixPrinter.__get_new_fig('Conf matrix default', fig_size)

        ax = sn.heatmap(cm_dataframe, annot=annotate, annot_kws={"size": font_size}, linewidths=line_width, ax=ax1,
                        cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=10) #Set ticklabels rotation
        ax.set_yticklabels(ax.get_yticklabels(), rotation=25, fontsize=10)

        # Turn off all the ticks
        for tick in ax.xaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)

        for tick in ax.yaxis.get_major_ticks():
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)

        quadmesh   = ax.findobj(QuadMesh)[0] #Face colours list
        facecolors = quadmesh.get_facecolors()

        array_df = np.array(cm_dataframe.to_records(index=False).tolist())
        text_add, text_del = [], []
        position = -1 #From left to right, bottom to top.
        for text in ax.collections[0].axes.texts:
            pos = np.array(text.get_position()) - [0.5,0.5]
            row, column = int(pos[1]), int(pos[0])
            position += 1

            #Set text
            txt_res = ConfusionMatrixPrinter.__configcell_text_and_colors(array_df, row, column, text, facecolors, position, font_size, fmt, show_null_values)
            text_add.extend(txt_res[0])
            text_del.extend(txt_res[1])

        #Remove the old ones
        for item in text_del:
            item.remove()
        #Append the new ones
        for item in text_add:
            ax.text(item['x'], item['y'], item['text'], **item['kw'])

        ax.set_title('Classification Confusion Matrix')
        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        plt.tight_layout() #Set layout slim
        plt.show()
