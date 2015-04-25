#include <ui/mainwindow.h>
#include <QtCore/QFile>
MainWindow::MainWindow(QMainWindow *parent)
    : QMainWindow(parent) 
{
    ui.setupUi(this);
    QFile File("src/ui/resources/cuda.stylesheet");
    File.open(QFile::ReadOnly);
    QString StyleSheet = QLatin1String(File.readAll());
    // this->setStyleSheet("background-image: url(src/ui/resources/background.jpg)");
	ui.ImageGB->setStyleSheet("background-image: url(src/ui/resources/background.jpg);");
	ui.splitter->setStretchFactor(0, 1);
	ui.splitter->setStretchFactor(1, 0);

    qApp->setStyleSheet(StyleSheet);
}