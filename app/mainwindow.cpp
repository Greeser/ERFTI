#include "mainwindow.h"
#include "ui_mainwindow.h"


//caffe::CaffeBinding* kCaffeBinding = new caffe::CaffeBinding();


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    screen_ = Screens::Start;
    connect(ui->start_page, SIGNAL(logged()), this, SLOT(log_in()));
    connect(ui->actionStart,SIGNAL(hovered()),ui->start_page,SLOT(start_stream()));
    int gpu_id = -1;
    std::string model_folder = "/home/greeser/Diplom/ERFTI/app/model";
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

void MainWindow::log_in()
{
    screen_ = Screens::Work;
    ui->stackedWidget->setCurrentIndex(static_cast<int>(screen_));
    ui->work_page->setNet(net_);
    ui->work_page->start_stream();

}
