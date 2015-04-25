#include "ui_base_window.h"
#include <QtWidgets/QMainWindow>
#ifndef __PARALLEL_IMAGE_MAINWINDOW_H__
#define __PARALLEL_IMAGE_MAINWINDOW_H__


class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	explicit MainWindow(QMainWindow *parent = 0);
	void closeEvent(QCloseEvent* e);

private:
	 Ui_MainWindow ui;
};

#endif