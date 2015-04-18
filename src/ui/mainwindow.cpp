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
    qApp->setStyleSheet(StyleSheet);
}