
from collections import namedtuple
import json
import html
from gamegym.algorithms.mcts import search


class TreeNode:

    __slots__ = ("value", "children", "title", "highlight")

    def __init__(self, value, title, highlight=False):
        self.value = value
        self.title = title
        self.highlight = highlight
        self.children = []

    def child(self, value, title, highlight):
        node = TreeNode(value, title, highlight)
        self.children.append(node)
        return node

    def traverse(self):
        yield self
        for child in self.children:
          yield from child.traverse()

    def to_dict(self, transform_value, transform_title):
        d = {
          "title": transform_title(self.title),
          "value": transform_value(self.value),
        }
        if self.highlight:
          d["highlight"] = True
        if self.children:
          d["children"] = [child.to_dict(transform_value, transform_title)
                           for child in self.children]
        return d

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Gamegym</title>
<script src="https://d3js.org/d3.v5.min.js"></script>
<style>

body {
  font: 10px sans-serif;
}

.link {
  fill: none;
  stroke: #888;
  stroke-width: 1;
}

.border {
  fill: none;
  shape-rendering: crispEdges;
  stroke: #aaa;
}

.node {
}

.nodetext {
  pointer-events: none;
  font-family: monospace;
  font-size: 15;
  text-anchor: middle;
}

.nodebox {
 fill: none;
 stroke-width: 1;
 stroke: black;");
}

.nodetitle {
  font-family: sans;
  font-size: 20;
  text-anchor: middle;
  pointer-events: none;
  fill: #020;
}

.titlebox {
 fill: #aba;
 stroke-width: 1;
 stroke: #020;");
}

.nodetitle_h {
  font-family: sans;
  font-size: 20;
  text-anchor: middle;
  pointer-events: none;
  fill: #200;
}

.titlebox_h {
 fill: #baa;
 stroke-width: 1;
 stroke: #200;");
}

</style>
</head>
<body>
<script type="text/javascript">

  function getTextWidth(text, font) {
      let canvas = document.createElement("canvas");
      var context = canvas.getContext("2d");
      context.font = font;
      var metrics = context.measureText(text);
      return metrics.width;
  }

const data = %DATA%;

// set the dimensions and margins of the diagram
var margin = {top: 40, right: 90, bottom: 50, left: 90},
    width = 660 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

const title_h = %TITLEHEIGHT% * 20;
const box_w = getTextWidth("W", getTextWidth("15 monospace")) * %BOXWIDTH% + 4;
const box_h = %BOXHEIGHT% * 15 + 10 + title_h;

// declares a tree layout and assigns the size
var treemap = d3.tree().nodeSize([box_w + 20, box_h + 50]);
    //.size([width, height]);

//  assigns the data to a hierarchy using parent-child relationships
var nodes = d3.hierarchy(data);

// maps the node data to the tree layout
nodes = treemap(nodes);

// append the svg obgect to the body of the page
// appends a 'group' element to 'svg'
// moves the 'group' element to the top left margin
var svg = d3.select("body").append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom),
    g = svg.append("g");
      /*.attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");*/

svg.append("rect")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .style("fill", "none")
    .style("pointer-events", "all")
    .call(d3.zoom()
        .scaleExtent([1 / 2, 4])
        .on("zoom", zoomed));

function zoomed() {
  g.attr("transform", d3.event.transform);
}

var link = g.selectAll(".link")
    .data( nodes.descendants().slice(1))
  .enter().append("path")
    .attr("class", "link")
    .attr("d", function(d) {
       return "M" + d.x + "," + d.y
         + "C" + d.x + "," + (d.y + d.parent.y + box_h) / 2
         + " " + d.parent.x + "," +  (d.y + d.parent.y + box_h) / 2
         + " " + d.parent.x + "," + (d.parent.y + box_h);
       });

var node = g.selectAll(".node")
    .data(nodes.descendants())
  .enter().append("g")
/*    .attr("class", function(d) {
      return "node" +
        (d.children ? " node--internal" : " node--leaf"); })*/
    .attr("transform", function(d) {
      return "translate(" + d.x + "," + d.y + ")"; });

node.append("rect")
  .attr("x", -box_w / 2)
  .attr("width", box_w)
  .attr("height", box_h)
  .attr("class", "nodebox");

node.append("text")
  .attr("class", "nodetext")
  .attr("y", title_h)
  .html(function(d) { return d.data.value; });

node.append("rect")
  .attr("x", -box_w / 2)
  .attr("width", box_w)
  .attr("height", title_h)
  .attr("class", function(d) { return d.data.highlight ? "titlebox_h" : "titlebox"})

node.append("text")
  .attr("y", -5)
  .attr("class", function(d) { return d.data.highlight ? "nodetitle_h" : "nodetitle"})
  .html(function(d) { return d.data.title; });


console.log(treemap.size());

</script>
</body>
</html>"""


def export_tree_to_html(root, output):
    maxw = 0
    maxh = 0
    titleh = 0
    for node in root.traverse():
      h = node.value.count("\n") + 1
      maxh = max(maxh, h)
      w = max(len(line) for line in node.value.split("\n"))
      maxw = max(maxw, w)
      h = node.title.count("\n") + 1
      titleh = max(titleh, h)

    html = HTML_TEMPLATE \
            .replace("%TITLEHEIGHT%", str(titleh)) \
            .replace("%BOXWIDTH%", str(maxw)) \
            .replace("%BOXHEIGHT%", str(maxh)) \
            .replace("%DATA%", json.dumps(root.to_dict(lambda t: text_to_svg(t, 15), lambda t: text_to_svg(t, 20))))

    with open(output, "w") as f:
        f.write(html)


def text_to_svg(text, height):
  return "".join("<tspan x=0 dy={}>{}</tspan>".format(height, line.replace(" ", "&#160;")) for i, line in enumerate(text.split("\n")))


def export_play_tree(play_info, output, max_actions=10, show_limit=1e-8):
    game = play_info.game
    adapter = game.TextAdapter(game)

    root = TreeNode(html.escape(adapter.get_observation(game.start()).data), "init", True)
    node = root

    for s, action, ds in play_info.replay():
        if ds is None:
          break
        items = sorted(ds.items(), key=lambda x: x[1], reverse=True)[:max_actions]
        items = list(filter(lambda x: x[1] >= show_limit, items))
        if action not in [k for k, v in items]:
            items.append((action, dict(ds.items())[action]))
        next_node = None
        for a, p in ds.items():
          s2 = game.play(s, a)
          n = node.child(html.escape(adapter.get_observation(s2).data), "{:0.3g}".format(p * 100), a == action)
          if a == action:
            next_node = n
        node = next_node
    export_tree_to_html(root, output)


def export_az_play_tree(az, output, num_simulations=None, max_actions=10, show_limit=1):
  game = az.game
  est = az.last_estimator()
  adapter = game.TextAdapter(game)

  if num_simulations is None:
    num_simulations = az.num_simulations

  def make_move(situation):
      return action,

  situation = game.start()
  root = TreeNode(html.escape(adapter.get_observation(situation).data), "init", True)
  node = root
  next_node = None
  next_situation = None
  while not situation.is_terminal():
      s = search.MctSearch(situation, est)
      s.search(num_simulations)
      action = s.best_action_max_visits()
      children = sorted(s.root.children.items(), key=lambda x: x[1].visit_count, reverse=True)[:max_actions]
      children = list(filter(lambda x: x[1].visit_count >= show_limit, children))

      first = True
      for a, sn in children:
          n = node.child(html.escape(adapter.get_observation(sn.situation).data), "{}\n{}".format(sn.visit_count, ",".join(["{:.1g}".format(v) for v in sn.value])), first)
          if first:
            next_node = n
            first = False
      node = next_node
      situation = children[0][1].situation
  export_tree_to_html(root, output)