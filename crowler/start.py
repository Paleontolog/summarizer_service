from scrapy.crawler import CrawlerProcess

from scrapper.crowler.spiders.crowl import MySpider

process = CrawlerProcess({
    'ITEM_PIPELINES': {'pipelines.pipel.JsonWithEncodingPipeline': 800},
    'DOWNLOAD_DELAY': 3
})

process.crawl(MySpider)
process.start()
