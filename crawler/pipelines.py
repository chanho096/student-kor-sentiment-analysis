# Define your item pipelines here
# 데이터 가공 및 DB저장을 수행하는 파일
# useful for handling different item types with a single interface
# export to csv
# from scrapy.exporters import CsvItemExporter
from __future__ import unicode_literals
import re

hangul = re.compile('[^ ㄱ-ㅣ가-힣|0-9|a-z|A-Z]+')

minLengthReview = 8
MAXGRADE = 7
MINGRADE = 4


class CrawlerPipeline:
    def process_item(self, item, spider):
        return item


class TextPipeline(object):
    count = 0
    list_csv = []
    bigTextGradeWraper = []

    """ export to csv
    def open_spider(self, spider):
        self.file = open("{0}.csv".format(spider.fileRangeName), 'wb')
        self.exporter = CsvItemExporter(self.file, encoding='utf-8')
        self.exporter.start_exporting()
    """

    def close_spider(self, spider):
        self.list_csv.append(self.bigTextGradeWraper)

    def process_item(self, item, spider):
        if self.count == 0:
            self.count = 1
            self.list_csv.append(str(item['reviewTitle'][0]))

        # Regulation
        for i in range(1, len(item['reviewText'])):
            item['reviewText'][0] = item['reviewText'][0] + item['reviewText'][i]
        item['reviewText'] = [item['reviewText'][0]]
        item['reviewText'][0] = re.sub(r"\n", " ", item['reviewText'][0], flags=re.UNICODE)
        item['reviewText'][0] = hangul.sub('', item['reviewText'][0])
        item['reviewText'][0] = re.sub(r"\s+", " ", item['reviewText'][0], flags=re.UNICODE)
        item['reviewText'][0] = item['reviewText'][0].strip(' ')

        # re-lavel
        if (len(item['reviewText'][0]) < 8) | ((int(item['reviewGrade'][0]) > MINGRADE) & (int(item['reviewGrade'][0]) < MAXGRADE)):
            pass
        else:
            if int(item['reviewGrade'][0]) < MAXGRADE:
                item['reviewGrade'][0] = 0
            else:
                item['reviewGrade'][0] = 1

            smallTextGradeWraper = [item['reviewText'][0], item['reviewGrade'][0]]
            self.bigTextGradeWraper.append(smallTextGradeWraper)

            # export to csv
            # self.exporter.export_item(item)
            return item

