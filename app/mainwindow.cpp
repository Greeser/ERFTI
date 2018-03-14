#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    screen = Screens::Start;
    connect(ui->start_page, SIGNAL(logged()), this, SLOT(log_in()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::log_in()
{
    screen = Screens::Work;
    ui->stackedWidget->setCurrentIndex(static_cast<int>(screen));
}
