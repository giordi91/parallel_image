#include "ui_base_window.h"
#include <QtWidgets/QMainWindow>
#include <core/filter_manager.h>
#include <QtCore/QString>

#ifndef __PARALLEL_IMAGE_MAINWINDOW_H__
#define __PARALLEL_IMAGE_MAINWINDOW_H__


class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QMainWindow *parent = 0);
	void closeEvent(QCloseEvent* e);

private:
	/**
	 * @brief This function is called whenever we need
	 * to refress the stack widgets status, we will clear 
	 * the stack and rebuild based on the new data.
	 */
	void update_stack_widgets();

private:
	 Ui_MainWindow ui;
	 Filter_manager * m_fm;

public slots:
	
	void open();
	void add_filter();

};

#endif