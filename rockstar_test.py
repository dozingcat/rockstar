import unittest

import rockstar

class RockstarText(unittest.TestCase):
    def testTokenize(self):
        t = rockstar.tokenize
        self.assertEqual(
            t(' hello   world '),
            ['hello', 'world'])
        self.assertEqual(
            t('tau is 6.28'),
            ['tau', 'is', '6.28'])
        self.assertEqual(
            t("Tommy ain't nobody"),
            ['Tommy', "ain't", 'nobody'])
        self.assertEqual(
            t('Alice said "Bob said \'yes\'"'),
            ['Alice', 'said', "Bob said 'yes'"])


if __name__ == '__main__':
    unittest.main()
