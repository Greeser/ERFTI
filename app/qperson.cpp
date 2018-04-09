#include "qperson.h"

#include "SimpleJSON/json.hpp"

#include <QDebug>
QPerson::QPerson(const QString &person_desc, const cv::Mat & image, int solution_version) : version_(solution_version)
{
    person_desc_ = person_desc;
    append_features(image);
}

QPerson::QPerson(int id, const QString &person_desc,
                 const QString &features_json, int solution_version) : person_id_(id), version_(solution_version)
{
    person_desc_ = person_desc;
    set_features_json(features_json.toStdString());
}

void QPerson::append_features(cv::Mat const & image)
{
    append_sample(image);
}

void QPerson::set_features_json(const std::string &json)
{
    features.clear();
    if (json.empty())
        return;

    try
    {
        auto obj = json::JSON::Load(json);

        if (obj.IsNull() || obj.JSONType() != json::JSON::Class::Array)
            return;

        for (auto & jfs : obj.ArrayRange())
        {
            std::vector<float> fs;
            for (auto & jf : jfs.ArrayRange())
                fs.push_back((float)jf.ToFloat());

            features.push_back(std::move(fs));
        }
    }
    catch (std::exception const &)
    {
    }
}

std::string QPerson::get_features_json() const
{
    auto obj = json::Array();

    for (auto & fs : features)
    {
        auto jfs = json::Array();

        for (auto f : fs)
        {
            jfs.append(f);
        }

        obj.append(jfs);
    }

    return obj.dump();
}

void QPerson::save_into_db(const QString &db_host, const QString &db_name, const QString &db_username, const QString &db_password)
{

    QSqlDatabase db = QSqlDatabase::addDatabase("QMYSQL");
    db.setHostName(db_host);
    db.setDatabaseName(db_name);
    db.setUserName(db_username);
    db.setPassword(db_password);
    bool ok = db.open();
   // QSqlQuery query;
    //query.exec("INSERT INTO persons (Name, SolutionVersion, KeyFeatures) VALUES (\"hh\",1, \"ggg\") ");

    QSqlQuery query;
    query.prepare("INSERT INTO persons (Name, SolutionVersion, KeyFeatures) VALUES (:Name, :SolVer, :KF)");
    query.bindValue(":Name", person_desc_);
    query.bindValue(":SolVer", version_);
    query.bindValue(":KF", QString::fromStdString(get_features_json()));
    query.exec();

    db.close();
}

void QPersonSet::load_from_sql(const QString &db_host, const QString &db_name, const QString &db_username, const QString &db_password)
{
    QSqlDatabase db = QSqlDatabase::addDatabase("QMYSQL");
    db.setHostName(db_host);
    db.setDatabaseName(db_name);
    db.setUserName(db_username);
    db.setPassword(db_password);
    bool ok = db.open();

    QSqlQuery query;
    query.exec("SELECT Id, Name, SolutionVersion, KeyFeatures FROM persons");
    while (query.next())
    {
        //auto person = std::make_shared<Person>(query.person_id, query.person_desc, query.key_features, query.solution_version);
        auto person = std::make_shared<QPerson>(query.value(0).toInt(), query.value(1).toString(), query.value(3).toString(), query.value(2).toInt());
        persons.push_back(person);
    }
    db.close();
    std::vector<std::shared_ptr<PersonFeatures>> ps;

    for (auto i = persons.begin(); i != persons.end();)
    {
        if (*i)
            ps.push_back(*i++);
        else
            i = persons.erase(i);
    }

    m_persons_features = std::move(PersonFeaturesSet(ps));

    qDebug () <<"Person loaded:" << persons.size();
}

std::vector<QPersonSet::QPersonPtr> QPersonSet::recognize(const cv::Mat &frame)
{
    try
    {
        FrameFeatures ff;
        ff.generate_features(frame);

        auto found = ff.compare_persons(m_persons_features);

        std::vector<std::shared_ptr<QPerson>> found2;
        found2.reserve(found.size());

        for (auto & p : found)
        {
            found2.push_back(std::static_pointer_cast<QPerson>(p));
        }

        return found2;
    }
    catch (...)
    {
        return{};
    }
}
