#include "signscreen.h"
#include "ui_signscreen.h"

SignScreen::SignScreen(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SignScreen)
{
    ui->setupUi(this);

    image_timer = new QTimer(this);
    connect(image_timer,SIGNAL(timeout()),this,SLOT(render_frame()));
}

SignScreen::~SignScreen()
{
    delete ui;
}

void SignScreen::start_stream()
{
    if( !mCapture.isOpened() )
            if( !mCapture.open( 0 ) )
                return;
    image_timer->setInterval(50);
    image_timer->start();
}

void SignScreen::render_frame()
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

void SignScreen::on_pushButton_clicked()
{
    cv::Mat frame;
    mCapture >> frame;
    auto name = ui->lineEdit->text();
    QPerson temp(name, frame, 0);
    temp.save_into_db("127.0.0.1", "ERFTI", "root", "root");
    image_timer->stop();
    mCapture.release();
    emit registered();
}
