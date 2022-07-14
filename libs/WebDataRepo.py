"""
    操作数据的基础工具类, 暂时仅封装tasks在DB以及本地目录的操作
                            cheungzeecn@gmail.com 2022-04-01
"""

import pymysql
import time
import logging
import os
# from confs import conf
# import kp_setup
import datetime


class WebDataRepo(object):
    def __init__(self, db_conf):
        self.db_conf = db_conf
        self.conn = pymysql.connect(**self.db_conf, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
        self.last_ping = time.time()

    def get_all_categories_ids(self):
        sql = 'SELECT * FROM categories'
        ret = self.query_db(sql, [])
        ret_dict = {}
        for rec in ret:
            ret_dict[rec['name_en']] = rec['id']
        return ret_dict

    def insert_category(self, name_en, desp_en='default', name_cn='', desp_cn='', created_at=None, updated_at=None):
        sql = ('INSERT into categories(name_en, desp_en, name_cn, desp_cn, created_at, updated_at) '
               'VALUES(%s, %s, %s, %s, %s, %s)')
        if created_at is None:
            created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if updated_at is None:
            updated_at = created_at
        return self.execute_db(sql, [name_en, desp_en, name_cn, desp_cn, created_at, updated_at])

    def insert_category2(self, name_en, desp_en='default', name_cn='', desp_cn='', created_at=None, updated_at=None):
        sql = ('INSERT into categories2(name_en, desp_en, name_cn, desp_cn, created_at, updated_at) '
               'VALUES(%s, %s, %s, %s, %s, %s)')
        if created_at is None:
            created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if updated_at is None:
            updated_at = created_at
        return self.execute_db(sql, [name_en, desp_en, name_cn, desp_cn, created_at, updated_at])

    def get_all_categories2_ids(self):
        sql = 'SELECT * FROM categories2'
        ret = self.query_db(sql, [])
        ret_dict = {}
        for rec in ret:
            ret_dict[rec['name_en']] = rec['id']
        return ret_dict

    def get_new_tasks_from_data_dir(self, data_dir):
        """
            遍历路径 发现新任务
        :param data_dir:
        :return: [(uuid, 路径), ..]
        """
        ret = []
        sub_dirs = os.listdir(data_dir)
        for sub_dir in sub_dirs:
            sub_dir_path = os.path.join(data_dir, sub_dir)
            for uuid in os.listdir(sub_dir_path):
                data_uri = os.path.join(sub_dir, uuid)
                article_dir = os.path.join(sub_dir_path, uuid)
                ret.append([uuid, data_uri, article_dir])
        return ret

    def insert_batch(self, batch_id, num):
        sql = ('insert into batchs(batch_id, num_all, num_done, status, msg, created_at, updated_at)'
                ' values(%s, %s, %s, %s, %s, %s, %s)')
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated_at = created_at
        return self.execute_db(sql, [batch_id, num, 0, 'new', 'init', created_at, updated_at])

    def update_batch(self, batch_id, num_all, num_done, status, msg):
        sql = "update batchs set num_all=%s, num_done=%s, status=%s, msg=%s, updated_at=%s where batch_id = %s"
        updated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.execute_db(sql, [num_all, num_done, status, msg, updated_at, batch_id])

    def auto_ping_conn(self, timeout=60):
        if time.time() - self.last_ping > timeout:
            self.conn.ping()
            self.last_ping = time.time()
        else:
            pass

    def update_aritcle_by_uuid(self, uuid, key_values: dict):
        if 'updated_at' not in key_values:
            key_values['updated_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_keys, all_values = zip(*key_values.items())
        all_key_str = ",".join(["{} = %s".format(k) for k in all_keys])

        sql = "UPDATE articles set {} where uuid=%s".format(all_key_str)
        return self.execute_db(sql, all_values + (uuid,))

    def get_article_by_uuid(self, uuid):
        sql = "SELECT * FROM articles WHERE uuid=%s"
        ret = self.query_db(sql, [uuid])
        if ret is not None:
            return ret[0]
        return None

    def insert_new_article_into_db(self, article):
        now = datetime.datetime.now()
        keys = list(article.keys())
        full_keys = keys + ['created_at', 'updated_at']
        keys_str = ", ".join(["`%s`" % k for k in full_keys])
        s_str = ", ".join(["%s"] * len(full_keys))
        keys_value_list = [article[k] for k in keys] + [str(now), str(now)]
        update_str = ", ".join(["`{}`=%s".format(k) for k in full_keys])
        sql = ("""INSERT INTO `articles` (%s) values (%s) on duplicate key """
               """update %s""") % (keys_str, s_str, update_str)
        try:
            with self.conn.cursor() as cursor:
                ret = cursor.execute(sql, keys_value_list*2)
                self.conn.commit()
                logging.info("insert article[%s] into DB: with ret[%s]" % (article['uuid'], ret))
        except Exception as e:
            logging.error("insert article[%s] to DB error, sql[%s][%s]" % (article['uuid'], sql, keys_value_list), exc_info=True)
            return False
        return True

    def delete_uuid_simi(self, uuid, method):
        sql = "DELETE from simi_articles WHERE `uuid`=%s AND `method`=%s"
        return self.execute_db(sql, [uuid, method])

    def insert_uuid_simi(self, uuid, simi_uuid, method, score, created_at=None):
        """
            一次一条, 稍微有点慢 以后再说吧 兜底数据不会太多
        :param uuid:
        :param simi_uuid:
        :param score:
        :param method:
        :param created_at:
        :return:
        """
        if created_at is None:
            created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = ("INSERT INTO simi_articles(uuid, simi_uuid, method, score, created_at) VALUES(%s, %s, %s, %s, %s) "\
              " ON DUPLICATE KEY UPDATE score=%s, created_at=%s")
        return self.execute_db(sql, [uuid, simi_uuid, method, score, created_at, score, created_at])

    def execute_db(self, sql, values):
        try:
            self.auto_ping_conn()
            with self.conn.cursor() as cursor:
                ret = cursor.execute(sql, values)
                self.conn.commit()
        except Exception as e:
            logging.error("execute_db error[{}]: sql[{}] values[{}]".format(e, sql, values), exc_info=True)
            return False
        return ret

    def query_db(self, sql, values):
        ret = None
        try:
            self.auto_ping_conn()
            with self.conn.cursor() as cursor:
                cursor.execute(sql, values)
                ret = cursor.fetchall()
                self.conn.commit()
        except Exception as e:
            logging.error("query_db error[{}]: sql[{}] values[{}]".format(e, sql, values), exc_info=True)
            return ret
        return ret

# if __name__ == '__main__':
#     data_repo = DataRepo(conf.db_conf, kp_setup.data_dir)
#     data_repo.update_task_by_uuid('de196834-ac21-11ec-a7a9-acde48001122', {'status': 'tmp'})
#     print(data_repo.get_task_by_uuid('de196834-ac21-11ec-a7a9-acde48001122')['status'])
#     data_repo.update_task_by_uuid('de196834-ac21-11ec-a7a9-acde48001122', {'status': 'new'})
