#include "workscreen.h"
#include "ui_workscreen.h"

WorkScreen::WorkScreen(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::WorkScreen)
{
    ui->setupUi(this);

    image_timer = new QTimer(this);
    connect(image_timer,SIGNAL(timeout()),this,SLOT(render_frame()));

    classifier_.reset(new EmotiW("/home/greeser/Diplom/ERFTI/app/emotion/net/deploy.prototxt",
                                 "/home/greeser/Diplom/ERFTI/app/emotion/net/EmotiW_VGG_S.caffemodel", -1));
}

WorkScreen::~WorkScreen()
{
    delete ui;
}

void WorkScreen::start_stream()
{
    if( !mCapture.isOpened() )
            if( !mCapture.open( 0 ) )
                return;
    image_timer->setInterval(50);
    image_timer->start();
}

void WorkScreen::render_frame()
{
    cv::Mat frame;
    mCapture >> frame;
    using Dim2Points = std::vector<std::vector<cv::Point2d>>;
    Dim2Points points;
    double min_face_size = 40;
    auto result = net_->GetDetection(frame,12 / min_face_size, 0.7, true, 0.7, true, points);
    for (auto & face : result)
    {
        cv::rectangle(frame, face.first, cv::Scalar(255,0,0), 4);
        current_frame_ = frame(face.first);
    }
    ui->inputWindow->showImage(frame);
}


void WorkScreen::on_pushButton_clicked()
{
    cv::resize(current_frame_, current_frame_, cv::Size(256,256));
    auto results = classifier_->GetEmotion(current_frame_);
    render_result(results);
}

void WorkScreen::render_result(EmAndConf &eac)
{
    for(auto& em : eac)
    {
        switch(em.first)
        {
        case Emotion::Angry:
            ui->angryBar->setValue(em.second * 100);
            break;
        case Emotion::Disgust:
            ui->disgustBar->setValue(em.second * 100);
            break;
        case Emotion::Fear:
            ui->fearBar->setValue(em.second * 100);
            break;
        case Emotion::Happy:
            ui->happyBar->setValue(em.second * 100);
            break;
        case Emotion::Neutral:
            ui->naturalBar->setValue(em.second * 100);
            break;
        case Emotion::Sad:
            ui->sadBar->setValue(em.second * 100);
            break;
        case Emotion::Surprise:
            ui->suprpiseBar->setValue(em.second * 100);
            break;
        }

    }
}
