#include <ui/mainwindow.h>
#include <QtCore/QFile>
#include <QtCore/QString>
#include <iostream>
#include <vector>
#include <string>

using std::vector;
using std::string;

MainWindow::MainWindow(QMainWindow *par)
    : QMainWindow(par), m_fm(nullptr) 
{
    ui.setupUi(this);
    QFile File("src/ui/resources/cuda.stylesheet");
    File.open(QFile::ReadOnly);
    QString StyleSheet = QLatin1String(File.readAll());
    qApp->setStyleSheet(StyleSheet);
    
    // this->setStyleSheet("background-image: url(src/ui/resources/background.jpg)");
	ui.ImageGB->setStyleSheet("background-image: url(src/ui/resources/background.jpg);");
	ui.splitter->setStretchFactor(0, 1);
	ui.splitter->setStretchFactor(1, 0);

	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(open()));
	connect(ui.filterPB, SIGNAL(clicked()), this, SLOT(add_filter()));

	//lets populate the combo box with the available filters
	vector<string> vec = Filter_manager::get_available_filters_name();
	ui.filterCB->clear();
	for (auto iter : vec )
	{
		ui.filterCB->addItem(QString(iter.c_str()));
	}
    ui.tex_w->hide();
}


void MainWindow::closeEvent(QCloseEvent* e)
{
	//forcing the widget to show in case was not shown before,
	//this will force the opengl to initialize and delete 
	//clean up properly
	ui.tex_w->show();
	e->accept();
}




void MainWindow::add_filter()
{
	QString filter_name = ui.filterCB->currentText();
    std::cout<<filter_name.toStdString()<<std::endl;
    if (!m_fm)
    {
    	return;
    }

    m_fm->add_filter_by_name(filter_name.toStdString().c_str());
    update_stack_widgets();

}


void MainWindow::open()
{
	//open the wanted image
	ui.tex_w->open();
	//initialize filter manager
	if (m_fm)
	{
		delete m_fm;
	}

	m_fm = new Filter_manager(ui.tex_w->get_image_data());

}

void update_stack_widgets()
{

	std::cout<<"updating stack widgets"<<std::endl;
}