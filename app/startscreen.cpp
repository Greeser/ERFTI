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
    using Dim2Points = std::vector<std::vector<cv::Point2d>>;
    Dim2Points points;
    double min_face_size = 40;
    auto result = net_->GetDetection(frame,12 / min_face_size, 0.7, true, 0.7, true, points);
    for (auto & face : result) {
        cv::rectangle(frame, face.first, cv::Scalar(255,0,0), 4);
    }
    ui->inputWindow->showImage(frame);
}
