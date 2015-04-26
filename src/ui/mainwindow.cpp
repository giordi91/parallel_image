#include <ui/mainwindow.h>
#include <QtCore/QFile>
#include <QtCore/QString>
#include <iostream>
#include <vector>
#include <string>
#include <ui/filter_widget.h>

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

void MainWindow::update_stack_widgets()
{

	std::cout<<"updating stack widgets"<<std::endl;
	std::cout<<"stack size "<<m_fm->stack_size()<<std::endl;

	clear_widgets_stack();
	size_t s_size = m_fm->stack_size();
	QWidget * w;
	m_filter_widgets.resize(s_size);
	for(size_t i=0; i<s_size; ++i)
	{
		w = new FilterWidget(ui.scrollAreaWidgetContents);
		ui.stack_VL->addWidget(w);
		m_filter_widgets[i] = w;  
	}

}

void MainWindow::clear_widgets_stack()
{
	for (auto w : m_filter_widgets)
	{
		delete(w);
	}
	m_filter_widgets.clear();
}