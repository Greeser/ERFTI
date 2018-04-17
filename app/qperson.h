#ifndef QPERSON_H
#define QPERSON_H
#include <QString>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlQuery>
#include <QVariant>
#include "frame_features.hpp"

#include <string>
#include <cstdint>
#include <map>
#include <vector>
#include <list>

class QPerson : public PersonFeatures
{
public:
    QPerson(QString const & person_desc, cv::Mat const & image, int solution_version);
    QPerson(int id, QString const & person_desc, QString const & features_json, int solution_version);
    
    void append_features(cv::Mat const & image);

    void set_features_json(std::string const & json);
    std::string get_features_json() const;
    
    void save_into_db(const QString &db_host, const QString &db_name, const QString &db_username, const QString &db_password);

public:
    int person_id_;
    QString person_desc_;
    int version_ = 0;

};

#endif // QPERSON_H


class QPersonSet
{
public:
    using QPersonPtr = std::shared_ptr<QPerson>;

    QPersonSet() = default;
    ~QPersonSet() = default;
    void load_from_sql(QString const & db_host, QString const & db_name, QString const & db_username, QString const & db_password);
    std::vector<QPersonPtr> recognize(cv::Mat const & frame);


public:

    std::list<QPersonPtr> persons;

private:

    PersonFeaturesSet m_persons_features;
};
