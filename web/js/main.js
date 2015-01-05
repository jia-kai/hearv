require.config({
  baseUrl: 'js',
  shim: {
    angularAMD: [
      'angular'
    ],
    'angular-route': [
      'angular'
    ],
    bootstrap: [
      'jquery'
    ],
    highcharts: {
      exports: 'Highcharts',
      deps: [
        'jquery'
      ]
    }
  },
  paths: {
    'ace-builds': '../bower_components/ace-builds/src-min/ace',
    bootstrap: '../bower_components/bootstrap/dist/js/bootstrap',
    jquery: '../bower_components/jquery/dist/jquery',
    requirejs: '../bower_components/requirejs/require',
    'prelude-amd': '../bower_components/prelude-amd/prelude',
    angularAMD: '../bower_components/angularAMD/angularAMD',
    angular: '../bower_components/angular/angular.min',
    'angular-route': '../bower_components/angular-route/angular-route',
    highcharts: '../bower_components/highcharts/highcharts',
    'highcharts-more': '../bower_components/highcharts/highcharts-more',
    exporting: '../bower_components/highcharts/modules/exporting',
    'angular-rangeslider': '../bower_components/angular-rangeslider/angular.rangeSlider'
  },
  packages: [

  ]
});

require(['main'], function() {
  require(['app'], function(app) {

  });
});
