#include "startscreen.h"
#include "ui_startscreen.h"

#include "QMessageBox"
StartScreen::StartScreen(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::StartScreen)
{
    ui->setupUi(this);

    image_timer = new QTimer(this);
    connect(image_timer,SIGNAL(timeout()),this,SLOT(render_frame()));
    persons_.load_from_sql("127.0.0.1", "ERFTI", "root", "root");
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
    cv::Mat frame;
    mCapture >> frame;
    auto results = persons_.recognize(frame);
    if (!results.empty())
    {
        std::shared_ptr<QPerson> person = results[0];
        image_timer->stop();
        mCapture.release();

        QMessageBox q;
        q.setText("You have logged as " + person->person_desc_);
        q.setStandardButtons(QMessageBox::Ok);
        q.setDefaultButton(QMessageBox::Ok);
        q.exec();

        emit logged(person->person_desc_);
    }
    else
    {
        QMessageBox q;
        q.setText("Some error during logging has occurred.");
        q.setStandardButtons(QMessageBox::Ok);
        q.setDefaultButton(QMessageBox::Ok);
        q.exec();
    }
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

void StartScreen::on_signUp_clicked()
{
    cv::Mat frame;
    mCapture >> frame;
    QPerson new_person("andriy", frame, 1);
    new_person.save_into_db("127.0.0.1", "ERFTI", "root", "root");
/*
    cv::Mat test = cv::imread("/home/greeser/Work/face_recognition/facenet/data/images/Anthony_Hopkins_0001.jpg");
    QPerson tp("Hopkins", test, 1);
    tp.save_into_db("127.0.0.1", "ERFTI", "root", "root");
*/
}
