P.when('A').register("ImageBlockATF", function(A){
    var data = {
                                'mbAapiDecision': true,
                                'asin' : 'B0TEST0001',
                                'colorImages': { 'initial': A.$.parseJSON('[{"hiRes":"https://m.media-amazon.com/images/I/TEST1._AC_SL1500_.jpg","thumb":"https://m.media-amazon.com/images/I/TEST1._AC_US40_.jpg","large":"https://m.media-amazon.com/images/I/TEST1._AC_.jpg","main":{"https://m.media-amazon.com/images/I/TEST1._AC_SX425_.jpg":[425,425]},"variant":"MAIN"},{"hiRes":null,"thumb":"https://m.media-amazon.com/images/I/TEST2._AC_US40_.jpg","large":"https://m.media-amazon.com/images/I/TEST2._AC_.jpg","variant":"PT01"},{"hiRes":"https://m.media-amazon.com/images/I/TEST3._AC_SL1500_.jpg","large":"https://m.media-amazon.com/images/I/TEST3._AC_.jpg","variant":"PT02"}]')},
                                'colorToAsin': {'initial': {}},
                                'heroImage': {},
                                'lVideoCount': 1,
                                'videos': A.$.parseJSON('[{"groupType":"IB_G1","durationSeconds":22,"title":"A Hub\'s \\"4K\\" output","url":"https://m.media-amazon.com/images/S/test-video/default.jobtemplate.mp4.480.mp4","slateUrl":"https://m.media-amazon.com/images/I/TESTSLATE.png","isVideo":true}]'),
                                'title': 'A made-up product'
    };
    A.trigger('P.AboveTheFold');
    return data;
});
