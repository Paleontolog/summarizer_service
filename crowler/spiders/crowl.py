from typing import List

from scrapper.crowler.data.object import Variant, Step
from scrapper.crowler.data import object as ob
from scrapy import Selector
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule


class MySpider(CrawlSpider):
    name = 'wiki_how_crowler'
    allowed_domains = ['ru.wikihow.com']

    custom_settings = {
        # 'DOWNLOAD_DELAY ': 3,
        'DOWNLOADER_MIDDLEWARES': {
            'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
            'scrapy_fake_useragent.middleware.RandomUserAgentMiddleware': 400,
        }
    }

    start_urls = ["https://ru.wikihow.com:Sitemap"]

    rules = (

        Rule(LinkExtractor(restrict_xpaths=("//a[div[@class='responsive_thumb_title']]",)),
             callback="parse_item", follow=True),

        Rule(LinkExtractor(restrict_xpaths=("//div[@class = 'pagination']//a",)),
             follow=True),

        Rule(LinkExtractor(restrict_xpaths=("//a[@rel='next' and contains(@class, 'button')]",)),
             follow=True),

        Rule(LinkExtractor(restrict_xpaths="//a[contains(@title, 'Category')]"), follow=True),
    )

    def join_texts(self, texts: List[List[str]]) -> str:
        return " <p> ".join(
            [" <p> ".join(text) for text in texts]
        )

    def parse_step(self, step_selector: Selector) -> Step:
        step = Step()
        step["number"] = \
            step_selector.xpath("//div[@class='step_num']/text()").extract_first()
        step["abstract"] = \
            step_selector.xpath("//div[@class='step']//b[@class='whb']/text()").extract_first()
        texts = [
            step_selector.xpath("//div[@class='step']/text()").extract(),
            step_selector.xpath("//div[@class='step']//li/text()").extract(),
        ]
        step["text"] = self.join_texts(texts)
        return step

    def parse_variant(self, variant_selector: Selector) -> Variant:
        variant = Variant()
        variant["variant_name"] = \
            variant_selector.xpath("//div[@class='altblock']/div").extract_first()
        steps = variant_selector.xpath("//li[contains(@id, 'step-id')]").extract()
        variant["steps"] = [self.parse_step(Selector(text=step)) for step in steps]
        return variant

    def get_variants(self, article_page: Selector) -> List[Variant]:
        variants = \
            article_page.xpath("//div[contains(@class, 'section steps')]").extract()
        variants = [
            self.parse_variant(Selector(text=variant)) for variant in variants
        ]
        return variants

    def parse_item(self, response):
        self.logger.info('This is an item page! %s', response.url)

        article_page = Selector(response)
        article = ob.Article()
        article["link"] = response.url
        article["main_title"] = \
            article_page.xpath("//h1//a/text()").extract_first()
        article["main_description"] = \
            article_page.xpath("//div[@class='mf-section-0']/p/text()").extract_first()
        article["advices"] = " <d> ".join(
            article_page.xpath("//div[contains(@id, 'advices')]//li/text()").extract()
        )
        article["warnings"] = " <d> ".join(
            article_page.xpath("//div[contains(@id, 'warnings')]//li/text()").extract()
        )

        article["variants"] = self.get_variants(article_page)

        yield article
