module.exports = (grunt) ->
  'use strict'

  require('load-grunt-tasks')(grunt)
  require('time-grunt')(grunt)

  config =
    app: 'app'
    build: 'build'
    dist: 'dist'

  grunt.initConfig
    config: config

    pkg: grunt.file.readJSON('package.json')

    coffee:
      dev:
        files: [
          expand: true
          cwd: 'js'
          src: ['**/*.coffee', '!**/_*.coffee']
          dest: '<%= config.build %>/js'
          ext: '.js'
        ]

    livescript:
      dev:
        files: [
          expand: true
          cwd: 'js'
          src: ['**/*.ls', '!**/_*.ls']
          dest: '<%= config.build %>/js'
          ext: '.js'
        ]

    sass:
      options:
        loadPath: [
          'bower_components'
        ]
      dev:
        files: [
          expand: true
          cwd: 'css'
          src: ['**/*.sass', '!**/_*.sass']
          dest: '<%= config.build %>/css'
          ext: '.css'
        ]

    autoprefixer:
      options:
        browsers: ['last 1 version', '> 1%']
      dev:
        files: [
          expand: true
          cwd: 'build/css'
          src: '*.css'
          dest: 'build/css'
        ]

    compass:
      dev:
        options:
          sassDir: 'sass'
          cssDir: 'build/css'

    slim:
      options:
        pretty: true
      dev:
        files: [
          expand: true
          src: ['*.slim', 'views/*.slim', '!_*.slim']
          dest: 'build'
          ext: '.html'
        ]

    compress:
      dev:
        options:
          archive: '/tmp/build.zip'
        files: [
          expand: true
          cwd: 'build'
          src: ['**']
          dest: 'build'
        ]

    clean:
      dev: 'build/*'

    connect:
      options:
        hostname: 'localhost'
        #hostname: '0.0.0.0'
        port: 9999
        livereload: true
        open: false
      livereload:
        options:
          base: '<%= config.build %>'
          middleware: (connect)->
            [
              connect.static(config.build)
              connect().use('/bower_components', connect.static('bower_components'))
              connect.static(config.app)
            ]

    copy:
      dev:
        files: [ {
          expand: true
          src: 'js/**/*.js'
          dest: 'build'
        }, {
          expand: true
          src: 'css/**/*.css'
          dest: 'build'
        }, {
          expand: true
          src: 'img/**/*'
          dest: 'build'
        }, {
          expand: true
          src: 'fonts/**/*'
          dest: 'build'
        } ]

    sprite:
      dev:
        src: 'sprites/*.png'
        destImg: 'build/img/sprites.png'
        destCSS: 'build/css/sprites.css'
        algorithm: 'binary-tree'
        padding: 0
        cssOpts:
          cssClass: (item) -> ".sprite-#{item.name}"

    browserify:
      dev:
        options:
          transform: ['coffeeify', 'debowerify']
        files: {}

    bower: # grunt-bower-requirejs
      dev:
        options:
          baseUrl: 'js'
        rjsConfig: 'js/main.js'

    notify:
      watch:
        options:
          title: 'grunt'
          message: 'changed'

    watch:
      options:
        livereload: true
        spawn: false

      livescript:
        files: ['js/**/*.ls']
        tasks: ['livescript:dev']

      coffee:
        files: ['js/**/*.coffee']
        tasks: ['coffee:dev']

      sass:
        files: ['css/**/*.sass']
        tasks: ['sass:dev', 'autoprefixer:dev']

      slim:
        files: ['*.slim', 'views/*.slim']
        #tasks: ['slim:dev']
        tasks: ['slim:dev']

  grunt.registerTask 'dev', ['bower:dev', 'copy:dev', 'livescript:dev', 'sass:dev', 'autoprefixer:dev', 'slim:dev'] # coffee
  grunt.registerTask 'default', ['dev', 'connect:livereload', 'watch']
