---
layout: default
title: Posts
permalink: /posts/
---


<h1>{{ page.title }}</h1>
<div id='posts' class='section'>
    {% for post in site.posts %}
        {% assign currentdate = post.date | date: "%Y" %}
        {% if currentdate != date %}
            {% unless forloop.first %}<br> </ul>{% endunless %}
            <p class='post-date-head' id="y{{post.date | date: "%Y"}}">{{ currentdate }}</p>
            <ul>
            {% assign date = currentdate %}
        {% endif %}
    <div class='post-row'>
        <p class='post-title'>
            <a href="{{ post.url }}">
                <h3>{{ post.title }}</h3>
            </a>
        </p>
        <p class='post-date'>
            {{ post.date | date_to_long_string }}
        </p>
    </div>
    <p class='post-subtitle'>
        {{ post.subtitle }}
    </p>
    {% endfor %}
<!-- </div> -->

