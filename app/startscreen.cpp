#include "startscreen.h"
#include "ui_startscreen.h"

StartScreen::StartScreen(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::StartScreen)
{
    ui->setupUi(this);

    image_timer = new QTimer(this);
    connect(image_timer,SIGNAL(timeout()),this,SLOT(render_frame()));


}

StartScreen::~StartScreen()
{
    delete ui;
}

void StartScreen::start_stream()
{
    if( !mCapture.isOpened() )
            if( !mCapture.open( 0 ) )
                return;
    image_timer->setInterval(50);
    image_timer->start();
}

void StartScreen::on_logIn_clicked()
{
    image_timer->stop();
    mCapture.release();
    emit logged();
}

void StartScreen::render_frame()
{
    cv::Mat frame;
    mCapture >> frame;
    ui->inputWindow->showImage(frame);
}
