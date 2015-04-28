#include <ui/mainwindow.h>
#include <QtCore/QFile>
#include <QtCore/QString>
#include <iostream>
#include <vector>
#include <string>
#include <ui/filter_widget.h>
#include <ui/int_widget.h>
#include <ui/float_widget.h>
#include <QtWidgets/QSpacerItem>


using std::vector;
using std::string;

MainWindow::MainWindow(QMainWindow *par)
    : QMainWindow(par), m_fm(nullptr), m_spacer(nullptr)
{
    ui.setupUi(this);
    QFile File("src/ui/resources/cuda.stylesheet");
    File.open(QFile::ReadOnly);
    QString StyleSheet = QLatin1String(File.readAll());
    qApp->setStyleSheet(StyleSheet);
    
	ui.splitter->setStretchFactor(0, 1);
	ui.splitter->setStretchFactor(1, 0);

	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(open()));
	connect(ui.filterPB, SIGNAL(clicked()), this, SLOT(add_filter()));
	connect(ui.compCB, SIGNAL(currentIndexChanged(int)), this, SLOT(set_computation(int)));

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
    update_image();
}


void MainWindow::open()
{
	//open the wanted image
	//initialize filter manager
	if (m_fm)
	{
		delete m_fm;
	}

	ui.tex_w->open();
	m_fm = new Filter_manager(ui.tex_w->get_image_data());
	update_stack_widgets();
}

void MainWindow::update_stack_widgets()
{

	// std::cout<<"updating stack widgets"<<std::endl;
	// std::cout<<"stack size "<<m_fm->stack_size()<<std::endl;
	//clearing the stack
	clear_widgets_stack();

	auto filter_data = m_fm->get_filters_data();
	FilterWidget * w ;

	for(auto filter_inst: filter_data)
	{
		w = generate_filter_widget(filter_inst);
		ui.stack_VL->addWidget(w);
		m_filter_widgets.push_back(w);  
	}

	m_spacer = new QSpacerItem(40, 20, QSizePolicy::Minimum, QSizePolicy::Expanding);
	ui.stack_VL->addSpacerItem(m_spacer);

}

FilterWidget * MainWindow::generate_filter_widget(Filter * filter_instance)
{
	FilterWidget * w = new FilterWidget(ui.scrollAreaWidgetContents);
	w->set_filter_name(filter_instance->get_type());

	for (auto attr : filter_instance->get_attributes())
	{
		// std::cout<<attr->type()<<" " <<attr->get_name()<<std::endl;
		std::string t = attr->type();
		if ( t == "m" )
		{
		    IntWidget * iw = new IntWidget(w,attr,this);
		    w->ui.mainVL->addWidget(iw);
			iw->ui.attributeL->setText((attr->get_name()+ ":").c_str());
		}
		else if (t == "f")
		{
		    FloatWidget * fw = new FloatWidget(w,attr);
			w->ui.mainVL->addWidget(fw);
			fw->ui.attributeL->setText((attr->get_name()+ ":").c_str());
		}


		
	}

	return w;
}


void MainWindow::clear_widgets_stack()
{
	for (auto w : m_filter_widgets)
	{
		ui.stack_VL->removeWidget(w);
		delete(w);
	}
	m_filter_widgets.clear();

	if (m_spacer)
	{
		ui.stack_VL->removeItem(m_spacer);
		delete m_spacer;
	}
}

void MainWindow::update_image()
{
    m_fm->evaluate_stack();
    ui.tex_w->upload_cpu_buffer(m_fm->get_output_buffer());

}

MainWindow::~MainWindow()
{
	//making sure to clear the widget stack
	clear_widgets_stack();
}

void MainWindow::set_computation(int comp_type)
{
	if (m_fm)
	{
		switch(comp_type)
		{
			case 0:
				m_fm->set_compute_type(Filter_manager::SERIAL);
				break;
			case 1:
				m_fm->set_compute_type(Filter_manager::TBB);
				break;			
			case 2:
				m_fm->set_compute_type(Filter_manager::CUDA);
				break;
		}
		

	}
}