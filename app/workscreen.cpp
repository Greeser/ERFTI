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
    ui->inputWindow->showImage(frame);
}

