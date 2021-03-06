#include "mainwindow.h"
#include "ui_mainwindow.h"


//caffe::CaffeBinding* kCaffeBinding = new caffe::CaffeBinding();


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    screen_ = Screens::Start;
    connect(ui->start_page, SIGNAL(logged(QString)), this, SLOT(log_in(QString)));
    connect(ui->start_page, SIGNAL(sign_up()), this, SLOT(sign_up()));
    connect(ui->sign_page, SIGNAL(registered()), this, SLOT(registered()));
    connect(ui->actionStart,SIGNAL(hovered()),ui->start_page,SLOT(start_stream()));
    int gpu_id = -1;
    std::string model_folder = "./model";
    net_.reset(new FaceInception::CascadeCNN(model_folder+"/det1-memory.prototxt", model_folder + "/det1.caffemodel",
                       model_folder + "/det1-memory-stitch.prototxt", model_folder + "/det1.caffemodel",
                       model_folder + "/det2-memory.prototxt", model_folder + "/det2.caffemodel",
                       model_folder + "/det3-memory.prototxt", model_folder + "/det3.caffemodel",
                       model_folder + "/det4-memory.prototxt", model_folder + "/det4.caffemodel",
                       gpu_id));

    ui->start_page->setNet(net_);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::log_in(QString name)
{
    screen_ = Screens::Work;
    ui->stackedWidget->setCurrentIndex(static_cast<int>(screen_));
    ui->work_page->setNet(net_);
    ui->work_page->start_stream();
    ui->work_page->set_name(name);

}

void MainWindow::sign_up()
{
    screen_ = Screens::Sign;
    ui->stackedWidget->setCurrentIndex(static_cast<int>(screen_));
    ui->sign_page->setNet(net_);
    ui->sign_page->start_stream();
}

void MainWindow::registered()
{
    screen_ = Screens::Start;
    ui->stackedWidget->setCurrentIndex(static_cast<int>(screen_));
    ui->start_page->updatePersons();
    ui->start_page->start_stream();

}
