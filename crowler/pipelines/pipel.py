import codecs
import json
import uuid
import re
from scrapy.utils.markup import remove_tags
from scrapy.utils.serialize import ScrapyJSONEncoder
from typing import Union

from scrapper.crowler.data.object import Variant, Step


class JsonWithEncodingPipeline(object):

    def __init__(self):
        self.file = \
            codecs.open(f'scraped_data_{uuid.uuid4()}.json',
                        'w',
                        encoding='utf-8')
        self.parsed_count = 0
        self.delete_spaces = re.compile(r"[\f\n\r\t\v]")

    def delete_tags(self, text: Union[str, None]) -> Union[str, None]:
        if text is not None:
            return remove_tags(text)
        else:
            return None

    def process_step(self, step: Step) -> Step:
        text = self.delete_spaces.sub("", step["text"])
        text = [chunk for chunk in text.split("<p>") if len(chunk) > 2]
        step["text"] = "<p>".join(text)
        return step

    def process_variant(self, variant: Variant) -> Variant:
        variant["variant_name"] = self.delete_tags(variant["variant_name"])
        variant["steps"] = [
            self.process_step(step) for step in variant["steps"]
        ]
        return variant

    def process_item(self, article, spider):
        if (self.parsed_count + 1) % 1000 == 0:
            self.file.close()
            self.file = \
                codecs.open(f'scraped_data_{uuid.uuid4()}.json',
                            'w',
                            encoding='utf-8')

        article["variants"] = [
            self.process_variant(variant) for variant in article["variants"]
        ]
        line = json.dumps(article, cls=ScrapyJSONEncoder) + "\n"
        self.file.write(line)
        self.parsed_count += 1
        return article

    def spider_closed(self, spider):
        self.file.close()
