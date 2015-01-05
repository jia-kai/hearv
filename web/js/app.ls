require <-! define 'app'
require! 'jquery'
require 'bootstrap'
require! 'highcharts'
window <<< require 'prelude-amd'
angularAMD <-!  require ['angularAMD', 'angular-route']

# {{{ highcharts options
Highcharts.setOptions do
  global:
    useUTC: false
  plotOptions:
    series:
      animation: false
      shadow: false
    line:
      marker:
        enabled: false
  legend:
    enabled: false
  exporting:
    buttons:
      contextButton:
        enabled: false
# }}}

#console.log jquery, map
#console.log angular

app = angular.module \app, [\ngRoute]

app.factory \DataService, ($http)->
  data = null
  promise = $http.get('all.json').success (d)!->
    data := d
  get = -> data
  { promise, get }

app.config ($routeProvider)!->
  $routeProvider
    .when '/',
      controller: \HomeCtrl
    .when '/examples/:example',
      controller: \ExampleCtrl
      templateUrl: 'views/example.html'

app.controller \ExampleCtrl, ($scope, $routeParams, DataService)!->
  $scope.$on '$viewContentLoaded', !->
    $('#video-modal').on 'hidden.bs.modal', !->
      $('video')[0].pause!
    $('#audio-modal').on 'hidden.bs.modal', !->
      $('audio')[0].pause!

  <-! DataService.promise.then

  data = DataService.get()
  $scope.example_name = $routeParams.example
  example = data[$scope.example_name]
  $scope.example = example
  prefix_sum = new Array example.frames.length+1
  t = 0
  for i til example.frames.length
    prefix_sum[i] = t
    t += example.frames[i].signal.length
  prefix_sum[prefix_sum.length-1] = t

  $scope.ix = 0
  $scope.is_opt = false
  curr_ix = 0
  curr_is_opt = false
  $scope.xslice = 2000
  $scope.yslice = 0

  $scope.fps = 10
  $scope.playing = false
  $scope.progress = 0
  update_progress = ->
    ix = +curr_ix
    v =
      try
        if curr_is_opt
          n = example.global.length-1
          if n > 0
            ix/n
          else
            0
        else
          n = example.frames.length
          if n > 0
            ix/n
          else
            0
      catch
        0
    $scope.progress = v*100
  $scope.min_yslice = ->
    Math.min $scope.max_yslice!, $scope.xslice
  $scope.max_yslice = ->
    if curr_is_opt
      example.global_config.time.length
    else
      prefix_sum[curr_ix]
  $scope.can_prev = ->
    $scope.ix > 0
  $scope.can_next = ->
    if curr_is_opt
      curr_ix < example.global.length-1
    else
      curr_ix < example.frames.length

  spectrum_chart = new Highcharts.Chart do
    chart:
      zoomType: \x
      renderTo: 'spectrum-canvas'
      animation: false
    plotOptions:
      series:
        color: '#880'
    credits: false
    title:
      text: ''
    xAxis:
      title:
        text: 'Frequency (Hz)'
      min: example.frame_config.spectrum_freq[0]
      max: last example.frame_config.spectrum_freq
    yAxis:
      title:
        text: 'Magnitude'
      min: example.frame_config.mag_range[0]
      max: example.frame_config.mag_range[1]
      showEmpty: true

  signal_chart = new Highcharts.Chart do
    chart:
      zoomType: \x
      renderTo: 'global-canvas'
      animation: false
    plotOptions:
      series:
        color: '#00c'
        states:
          hover:
            enabled: false
    tooltip:
      enabled: false
    credits: false
    title:
      text: ''
    rangeSelector:
      buttons: []
    scrollbar:
      enabled: true
    xAxis:
      title:
        text: 'Time (second)'
    yAxis:
      title:
        text: 'Amplitude'
      min: example.global_config.signal_range[0]
      max: example.global_config.signal_range[1]
    series: [ name: 'Signal', data: [] ]
  signal_plot = signal_chart.series[0]

  show_spectrum = (is_opt, ix)->
    if spectrum_chart.series.length > 0
      spectrum_chart.series[0].remove()
    if ! is_opt
      data =
        if ix == 0
          []
        else
          zip example.frame_config.spectrum_freq, example.frames[ix-1].spectrum
      plot = spectrum_chart.addSeries do
        name: 'Spectrum'
        data: data
      spectrum_chart.redraw!

  rebuild_signal = (ix, left, right)->
    f = (pos)->
      for i til ix
        if pos <= 0
          return [i, 0]
        t = example.frames[i].signal.length
        if (pos -= t) < 0
          return [i, pos+t]
      [ix, 0]

    [i0,j0] = f left
    [i1,j1] = f right
    data = []
    if i0 < i1
      signals = example.frames[i0].signal
      # padding
      data.push [signals[j0][0]-1e-6,0]
      for j from j0 til signals.length
        data.push signals[j]
      # padding
      data.push [last(signals)[0]+1e-6,0]
      for i from i0+1 til i1
        signals = example.frames[i].signal
        # padding
        data.push [signals[0][0]-1e-6,0]
        for j til signals.length
          data.push signals[j]
        # padding
        data.push [last(signals)[0]+1e-6,0]
    if i1 < ix
      signals = example.frames[i1].signal
      t = if i0 == i1 then j0 else 0
      # padding
      data.push [signals[t][0]-1e-6,0]
      for j from t til j1
        data.push signals[j]
      # padding
      data.push [last(signals)[0]+1e-6,0]
    signal_plot.setData data

  rebuild_global = ->
    ix = +$scope.ix
    times = example.global_config.time
    signals = example.global[ix].signal
    len = $scope.yslice
    data = []
    for j from Math.max(0, len-$scope.xslice) til len
      data.push [times[j], signals[j]]
    signal_plot.setData data

  show_signal = (is_opt, ix)->
    if is_opt
      rebuild_global!
    else
      if curr_ix < ix && ix < curr_ix+5
        for i from curr_ix til ix
          signals = example.frames[i].signal
          shift = signal_plot.data.length >= $scope.xslice
          signal_plot.addPoint [signals[0][0]-1e-6,0], false, shift
          for sig in signals
            shift = signal_plot.data.length >= $scope.xslice
            signal_plot.addPoint sig, false, shift
          shift = signal_plot.data.length >= $scope.xslice
          signal_plot.addPoint [last(signals)[0]+1e-6,0], false, shift
      else
        #for i from curr_ix-1 to ix by -1
        #  len = signal_plot.data.length
        #  for j til example.frames[i].signal.length
        #    signal_plot.data[len-1-j].remove false, false
        len = prefix_sum[ix]
        rebuild_signal ix, len-$scope.xslice, len
    signal_chart.redraw!

  normalize_yslice = !->
    if $scope.xslice > $scope.yslice
      $scope.yslice = $scope.xslice
    else
      t =
        if $scope.is_opt
          example.global_config.time.length
        else
          prefix_sum[$scope.ix]
      if t < $scope.yslice
        $scope.yslice = t

  $scope.set_ix = !->
    #console.log 'set_ix', curr_is_opt, $scope.is_opt, curr_ix, +$scope.ix
    is_opt = $scope.is_opt
    ix = +$scope.ix
    if is_opt
      normalize_yslice!
    else
      $scope.yslice = prefix_sum[ix]
    show_spectrum is_opt, ix
    show_signal is_opt, ix
    curr_is_opt := is_opt
    curr_ix := ix
    update_progress!

  $scope.set_slice = !->
    $scope.xslice = +$scope.xslice
    $scope.yslice = +$scope.yslice
    normalize_yslice!
    if curr_is_opt
      rebuild_global!
    else
      rebuild_signal $scope.ix, $scope.yslice-$scope.xslice, $scope.yslice
    signal_chart.redraw!


  timer = null

  play = !->
    console.log 'play'
    $scope.playing = true
    timer := setInterval tick, 1000/$scope.fps

  pause = !->
    console.log 'pause'
    $scope.playing = false
    if timer?
      clearInterval timer
      timer := null

  $scope.toggle = !->
    if $scope.playing
      pause!
    else
      play!

  $scope.prev = !->
    if curr_ix > 0
      $scope.ix = curr_ix-1
      if ! curr_is_opt
        $scope.yslice = prefix_sum[$scope.ix]
      $scope.set_ix!

  $scope.next = !->
    if $scope.can_next!
      $scope.ix = curr_ix+1
      if ! curr_is_opt
        $scope.yslice = prefix_sum[$scope.ix]
      $scope.set_ix!

  tick = !->
    if curr_is_opt
      if curr_ix >= example.global.length-1
        pause!
      else
        $scope.ix = curr_ix+1
        $scope.set_ix!
    else
      if curr_ix >= example.frames.length
        $scope.is_opt = true
        $scope.ix = 0
        $scope.yslice = example.global_config.time.length
        $scope.set_ix!
        pause!
      else
        $scope.ix = curr_ix+1
        $scope.yslice = prefix_sum[$scope.ix]
        $scope.set_ix!
    update_progress!
    $scope.$digest()

  do $scope.reset = !->
    pause!
    $scope.is_opt = false
    curr_is_opt := false
    $scope.ix = 0
    curr_ix := 0
    $scope.yslice = $scope.min_yslice()
    show_spectrum false, 0
    show_signal false, 0
    update_progress!

  $scope.to_opt = !->
    pause!
    $scope.is_opt = true
    curr_is_opt := true
    $scope.ix = 0
    curr_ix := 0
    $scope.yslice = $scope.max_yslice()
    show_spectrum true, 0
    show_signal true, 0
    update_progress!

app.controller 'HomeCtrl', ($scope, DataService)!->
  DataService.promise.then !->
    data = DataService.get()
    $('.loading').hide()
    $scope.examples = keys data
  , !->
    $('.glyphicon-refresh').removeClass!.addClass 'glyphicon glyphicon-remove'

angularAMD.bootstrap app

# vim: set fdm=marker
