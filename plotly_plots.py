import plotly
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import numpy as np

def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))

def plot_multimodel_stats(trainers, test_histories):
	epoch = np.arange(1,len(test_histories[0]['target_acc'])+1)
	trace_acc = [go.Scatter(
		x = epoch,
		y = test_histories[i]['target_acc'],
		name = 'model {}'.format(i),
		showlegend= False
	) for i in range(len(test_histories))]

	trace_source_acc = [go.Scatter(
		x = epoch,
		y = test_histories[i]['source_acc'],
		name = 'model {}'.format(i),
		showlegend= False
	) for i in range(len(test_histories))]

	trace_domain_acc = [go.Scatter(
		x = epoch,
		y = test_histories[i]['domain_acc'],
		name = 'model {}'.format(i),
		showlegend= False
	) for i in range(len(test_histories))]

	trace_loss = [go.Scatter(
		x = np.arange(len(trainers[i].train_history['train_loss'])),
		y = trainers[i].train_history['train_loss'],
		name = 'model {}'.format(i),
		showlegend= False
	) for i in range(len(trainers))]

	trace_len = [go.Scatter(
		x = np.arange(len(trainers[i].train_history['avg_len_c'])),
		y = np.array(trainers[i].train_history['avg_len_c']) + np.array(trainers[i].train_history['avg_len_d']),
		name = 'model {}'.format(i),
		showlegend= False
	) for i in range(len(trainers))]

	trace_dot = [go.Scatter(
		x = np.arange(len(trainers[i].train_history['avg_dot'])),
		y = trainers[i].train_history['avg_dot'],
		name = 'model {}'.format(i),
		showlegend= False
	) for i in range(len(trainers))]

	fig = tools.make_subplots(rows=3, cols=2, subplot_titles=('Target Domain Accuracy', 'Source Domain Accuracy', 'Domain Predictor Accuracy', 'Training Loss', 'Sum of avg len of vectors', 'Avg Dot Product'))
	for (t_acc, t_source, t_domain, t_loss, t_len, t_dot) in list(zip(trace_acc, trace_source_acc, trace_domain_acc, trace_loss, trace_len, trace_dot)):
		fig.append_trace(t_acc, 1, 1)
		fig.append_trace(t_source, 1, 2)
		fig.append_trace(t_domain, 2, 1)
		fig.append_trace(t_loss, 2, 2)
		fig.append_trace(t_len, 3, 1)
		fig.append_trace(t_dot, 3, 2)
		

	configure_plotly_browser_state()

	init_notebook_mode(connected=False)

	fig['layout'].update(height=800, width=1200, title='Models Test History')
	plotly.offline.iplot(fig, filename='basic-line')

def plot_domain_vanishing(trainers, test_histories, domain_histories, domain_gr_histories):
	
	epoch = np.arange(1,len(test_histories[0]['target_acc'])+1)
	
	trace_acc = [go.Scatter(
		x = epoch,
		y = test_histories[i]['target_acc'],
		name = 'model {}'.format(i),
		showlegend= False
	) for i in range(len(test_histories))]

	trace_source_acc = [go.Scatter(
		x = epoch,
		y = test_histories[i]['source_acc'],
		name = 'model {}'.format(i),
		showlegend= False
	) for i in range(len(test_histories))]

	trace_domain_model_c = [go.Scatter(
		x = epoch,
		y = domain_histories[i]['acc'],
		name = 'model {}'.format(i),
		showlegend= False
	) for i in range(len(test_histories))]

	trace_domain_model_f = [go.Scatter(
		x = epoch,
		y = domain_gr_histories[i]['acc'],
		name = 'model {}'.format(i),
		showlegend= False
	) for i in range(len(test_histories))]

	fig = tools.make_subplots(rows=2, cols=2, subplot_titles=('Target Domain Accuracy', 'Source Domain Accuracy', 'Domain Predictor on Class Predictor Accuracy', 'Domain Predictor on Feature Extractor Accuracy'))
	for (t_acc, t_source, t_dom_c, t_dom_f) in list(zip(trace_acc, trace_source_acc, trace_domain_model_c, trace_domain_model_f)):
		fig.append_trace(t_acc, 1, 1)
		fig.append_trace(t_source, 1, 2)
		fig.append_trace(t_dom_c, 2, 1)
		fig.append_trace(t_dom_f, 2, 2)
		

	configure_plotly_browser_state()

	init_notebook_mode(connected=False)

	fig['layout'].update(height=800, width=1200, title='Models Test History')
	plotly.offline.iplot(fig, filename='basic-line')