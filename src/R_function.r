# ------------------------------------------------------------------------------------
# 번  호: 1
# 함수명: make_lag_data_fn
# 입력값: data(변수선택 할 데이터), y(종속변수), times(지연(lag)할 시점)
# 설  명: y변수를 제외한 모든 변수를  times만큼 시점을 뒤로 이동시켜 컬럼 생성
# ------------------------------------------------------------------------------------

make_lag_data_fn <- function(data, y, times){
  input_df <- as.data.table(data)
  X <- data.table(input_df[, y, with = F], # 종속변수(y)의 열만 추출
                  input_df[, -1][, shift(.SD, 24, NA, 'lag', TRUE), .SDcols = 1:(ncol(input_df)-1)]) 
                  # 변수별 times개수만큼 lag 생성
                  # .SD:벡터(원본데이터)(.SD는 그룹핑 컬럼을 제외한 모든 칼럼을 의미함)
                  # 1:times: 몇번째 이전(이후) 값을 가져올건지 설정
                  # NA: 채울값이 없을경우 default값 설정
                  # lag: 이전 값을 가져올지 이후 값을 가져올지 type설정, True: 값이름 지정유무
                  # .SDcols1: 연산대상이 되는 특정컬럼을 지정

  return(X)
}


# ------------------------------------------------------------------------------------
# 번  호: 2
# 함수명: plot_nnet_fn
# 입력값: x(class 'nn' 인 객체), ...
# 설  명: neuralnet()함수에 의해 생성되는 class 'nn' 객체를 위한 시각화
# ------------------------------------------------------------------------------------

# github에서 소스 가져오기
# 경로: https://gist.githubusercontent.com/Peque/41a9e20d6687f2f3108d/raw/85e14f3a292e126f1454864427e3a189c2fe33f3/nnet_plot_update.r

plot_nnet_fn <- function(mod.in,nid=T,all.out=T,all.in=T,bias=T,wts.only=F,rel.rsc=5,
                    circle.cex=5,node.labs=T,var.labs=T,x.lab=NULL,y.lab=NULL,
                    line.stag=NULL,struct=NULL,cex.val=1,alpha.val=1,
                    circle.col='lightblue',pos.col='black',neg.col='grey',
                    bord.col='black', ...){
  require(scales)
  
  #sanity checks
  if('mlp' %in% class(mod.in)) warning('Bias layer not applicable for rsnns object')
  if('numeric' %in% class(mod.in)){
    if(is.null(struct)) stop('Three-element vector required for struct')
    if(length(mod.in) != ((struct[1]*struct[2]+struct[2]*struct[3])+(struct[3]+struct[2])))
      stop('Incorrect length of weight matrix for given network structure')
  }
  if('train' %in% class(mod.in)){
    if('nnet' %in% class(mod.in$finalModel)){
      mod.in<-mod.in$finalModel
      warning('Using best nnet model from train output')
    }
    else stop('Only nnet method can be used with train object')
  }
  
  #gets weights for neural network, output is list
  #if rescaled argument is true, weights are returned but rescaled based on abs value
  nnet.vals<-function(mod.in,nid,rel.rsc,struct.out=struct){
    
    require(scales)
    require(reshape)
    
    if('numeric' %in% class(mod.in)){
      struct.out<-struct
      wts<-mod.in
    }
    
    #neuralnet package
    if('nn' %in% class(mod.in)){
      struct.out<-unlist(lapply(mod.in$weights[[1]],ncol))
    	struct.out<-struct.out[-length(struct.out)]
    	struct.out<-c(
    		length(mod.in$model.list$variables),
    		struct.out,
    		length(mod.in$model.list$response)
    		)    		
      wts<-unlist(mod.in$weights[[1]])   
    }
    
    #nnet package
    if('nnet' %in% class(mod.in)){
      struct.out<-mod.in$n
      wts<-mod.in$wts
    }
    
    #RSNNS package
    if('mlp' %in% class(mod.in)){
      struct.out<-c(mod.in$nInputs,mod.in$archParams$size,mod.in$nOutputs)
      hid.num<-length(struct.out)-2
      wts<-mod.in$snnsObject$getCompleteWeightMatrix()
      
      #get all input-hidden and hidden-hidden wts
      inps<-wts[grep('Input',row.names(wts)),grep('Hidden_2',colnames(wts)),drop=F]
      inps<-melt(rbind(rep(NA,ncol(inps)),inps))$value
      uni.hids<-paste0('Hidden_',1+seq(1,hid.num))
      for(i in 1:length(uni.hids)){
        if(is.na(uni.hids[i+1])) break
        tmp<-wts[grep(uni.hids[i],rownames(wts)),grep(uni.hids[i+1],colnames(wts)),drop=F]
        inps<-c(inps,melt(rbind(rep(NA,ncol(tmp)),tmp))$value)
        }
      
      #get connections from last hidden to output layers
      outs<-wts[grep(paste0('Hidden_',hid.num+1),row.names(wts)),grep('Output',colnames(wts)),drop=F]
      outs<-rbind(rep(NA,ncol(outs)),outs)
      
      #weight vector for all
      wts<-c(inps,melt(outs)$value)
      assign('bias',F,envir=environment(nnet.vals))
      }
    
    if(nid) wts<-rescale(abs(wts),c(1,rel.rsc))
    
    #convert wts to list with appropriate names 
    hid.struct<-struct.out[-c(length(struct.out))]
    row.nms<-NULL
    for(i in 1:length(hid.struct)){
      if(is.na(hid.struct[i+1])) break
      row.nms<-c(row.nms,rep(paste('hidden',i,seq(1:hid.struct[i+1])),each=1+hid.struct[i]))
    }
    row.nms<-c(
      row.nms,
      rep(paste('out',seq(1:struct.out[length(struct.out)])),each=1+struct.out[length(struct.out)-1])
      )
    out.ls<-data.frame(wts,row.nms)
    out.ls$row.nms<-factor(row.nms,levels=unique(row.nms),labels=unique(row.nms))
    out.ls<-split(out.ls$wts,f=out.ls$row.nms)
    
    assign('struct',struct.out,envir=environment(nnet.vals))
    
    out.ls
    
    }
  
  wts<-nnet.vals(mod.in,nid=F)
  
  if(wts.only) return(wts)
  
  #circle colors for input, if desired, must be two-vector list, first vector is for input layer
  if(is.list(circle.col)){
                    circle.col.inp<-circle.col[[1]]
                    circle.col<-circle.col[[2]]
                    }
  else circle.col.inp<-circle.col
  
  #initiate plotting
  x.range<-c(0,100)
  y.range<-c(0,100)
  #these are all proportions from 0-1
  if(is.null(line.stag)) line.stag<-0.011*circle.cex/2
  layer.x<-seq(0.17,0.9,length=length(struct))
  bias.x<-layer.x[-length(layer.x)]+diff(layer.x)/2
  bias.y<-0.95
  circle.cex<-circle.cex
  
  #get variable names from mod.in object
  #change to user input if supplied
  if('numeric' %in% class(mod.in)){
    x.names<-paste0(rep('X',struct[1]),seq(1:struct[1]))
    y.names<-paste0(rep('Y',struct[3]),seq(1:struct[3]))
  }
  if('mlp' %in% class(mod.in)){
    all.names<-mod.in$snnsObject$getUnitDefinitions()
    x.names<-all.names[grep('Input',all.names$unitName),'unitName']
    y.names<-all.names[grep('Output',all.names$unitName),'unitName']
  }
  if('nn' %in% class(mod.in)){
    x.names<-mod.in$model.list$variables
    y.names<-mod.in$model.list$respons
  }
  if('xNames' %in% names(mod.in)){
    x.names<-mod.in$xNames
    y.names<-attr(terms(mod.in),'factor')
    y.names<-row.names(y.names)[!row.names(y.names) %in% x.names]
  }
  if(!'xNames' %in% names(mod.in) & 'nnet' %in% class(mod.in)){
    if(is.null(mod.in$call$formula)){
      x.names<-colnames(eval(mod.in$call$x))
      y.names<-colnames(eval(mod.in$call$y))
    }
    else{
      forms<-eval(mod.in$call$formula)
      x.names<-mod.in$coefnames
      facts<-attr(terms(mod.in),'factors')
      y.check<-mod.in$fitted
      if(ncol(y.check)>1) y.names<-colnames(y.check)
      else y.names<-as.character(forms)[2]
    } 
  }
  #change variables names to user sub 
  if(!is.null(x.lab)){
    if(length(x.names) != length(x.lab)) stop('x.lab length not equal to number of input variables')
    else x.names<-x.lab
  }
  if(!is.null(y.lab)){
    if(length(y.names) != length(y.lab)) stop('y.lab length not equal to number of output variables')
    else y.names<-y.lab
  }
  
  #initiate plot
  plot(x.range,y.range,type='n',axes=F,ylab='',xlab='',...)
  
  #function for getting y locations for input, hidden, output layers
  #input is integer value from 'struct'
  get.ys<-function(lyr){
    spacing<-diff(c(0*diff(y.range),0.9*diff(y.range)))/max(struct)
    seq(0.5*(diff(y.range)+spacing*(lyr-1)),0.5*(diff(y.range)-spacing*(lyr-1)),
        length=lyr)
  }
  
  #function for plotting nodes
  #'layer' specifies which layer, integer from 'struct'
  #'x.loc' indicates x location for layer, integer from 'layer.x'
  #'layer.name' is string indicating text to put in node
  layer.points<-function(layer,x.loc,layer.name,cex=cex.val){
    x<-rep(x.loc*diff(x.range),layer)
    y<-get.ys(layer)
    points(x,y,pch=21,cex=circle.cex,col=bord.col,bg=in.col)
    if(node.labs) text(x,y,paste(layer.name,1:layer,sep=''),cex=cex.val)
    if(layer.name=='I' & var.labs) text(x-line.stag*diff(x.range),y,x.names,pos=2,cex=cex.val)      
    if(layer.name=='O' & var.labs) text(x+line.stag*diff(x.range),y,y.names,pos=4,cex=cex.val)
  }
  
  #function for plotting bias points
  #'bias.x' is vector of values for x locations
  #'bias.y' is vector for y location
  #'layer.name' is  string indicating text to put in node
  bias.points<-function(bias.x,bias.y,layer.name,cex,...){
    for(val in 1:length(bias.x)){
      points(
        diff(x.range)*bias.x[val],
        bias.y*diff(y.range),
        pch=21,col=bord.col,bg=in.col,cex=circle.cex
      )
      if(node.labs)
        text(
          diff(x.range)*bias.x[val],
          bias.y*diff(y.range),
          paste(layer.name,val,sep=''),
          cex=cex.val
        )
    }
  }
  
  #function creates lines colored by direction and width as proportion of magnitude
  #use 'all.in' argument if you want to plot connection lines for only a single input node
  layer.lines<-function(mod.in,h.layer,layer1=1,layer2=2,out.layer=F,nid,rel.rsc,all.in,pos.col,
                        neg.col,...){
    
    x0<-rep(layer.x[layer1]*diff(x.range)+line.stag*diff(x.range),struct[layer1])
    x1<-rep(layer.x[layer2]*diff(x.range)-line.stag*diff(x.range),struct[layer1])
    
    if(out.layer==T){
      
      y0<-get.ys(struct[layer1])
      y1<-rep(get.ys(struct[layer2])[h.layer],struct[layer1])
      src.str<-paste('out',h.layer)
      
      wts<-nnet.vals(mod.in,nid=F,rel.rsc)
      wts<-wts[grep(src.str,names(wts))][[1]][-1]
      wts.rs<-nnet.vals(mod.in,nid=T,rel.rsc)
      wts.rs<-wts.rs[grep(src.str,names(wts.rs))][[1]][-1]
      
      cols<-rep(pos.col,struct[layer1])
      cols[wts<0]<-neg.col
      
      if(nid) segments(x0,y0,x1,y1,col=cols,lwd=wts.rs)
      else segments(x0,y0,x1,y1)
      
    }
    
    else{
      
      if(is.logical(all.in)) all.in<-h.layer
      else all.in<-which(x.names==all.in)
      
      y0<-rep(get.ys(struct[layer1])[all.in],struct[2])
      y1<-get.ys(struct[layer2])
      src.str<-paste('hidden',layer1)
      
      wts<-nnet.vals(mod.in,nid=F,rel.rsc)
      wts<-unlist(lapply(wts[grep(src.str,names(wts))],function(x) x[all.in+1]))
      wts.rs<-nnet.vals(mod.in,nid=T,rel.rsc)
      wts.rs<-unlist(lapply(wts.rs[grep(src.str,names(wts.rs))],function(x) x[all.in+1]))
      
      cols<-rep(pos.col,struct[layer2])
      cols[wts<0]<-neg.col
      
      if(nid) segments(x0,y0,x1,y1,col=cols,lwd=wts.rs)
      else segments(x0,y0,x1,y1)
      
    }
    
  }
  
  bias.lines<-function(bias.x,mod.in,nid,rel.rsc,all.out,pos.col,neg.col,...){
    
    if(is.logical(all.out)) all.out<-1:struct[length(struct)]
    else all.out<-which(y.names==all.out)
    
    for(val in 1:length(bias.x)){
      
      wts<-nnet.vals(mod.in,nid=F,rel.rsc)
      wts.rs<-nnet.vals(mod.in,nid=T,rel.rsc)
      
    	if(val != length(bias.x)){
        wts<-wts[grep('out',names(wts),invert=T)]
        wts.rs<-wts.rs[grep('out',names(wts.rs),invert=T)]
    		sel.val<-grep(val,substr(names(wts.rs),8,8))
    		wts<-wts[sel.val]
    		wts.rs<-wts.rs[sel.val]
    		}
    
    	else{
        wts<-wts[grep('out',names(wts))]
        wts.rs<-wts.rs[grep('out',names(wts.rs))]
      	}
      
      cols<-rep(pos.col,length(wts))
      cols[unlist(lapply(wts,function(x) x[1]))<0]<-neg.col
      wts.rs<-unlist(lapply(wts.rs,function(x) x[1]))
      
      if(nid==F){
        wts.rs<-rep(1,struct[val+1])
        cols<-rep('black',struct[val+1])
      }
      
      if(val != length(bias.x)){
        segments(
          rep(diff(x.range)*bias.x[val]+diff(x.range)*line.stag,struct[val+1]),
          rep(bias.y*diff(y.range),struct[val+1]),
          rep(diff(x.range)*layer.x[val+1]-diff(x.range)*line.stag,struct[val+1]),
          get.ys(struct[val+1]),
          lwd=wts.rs,
          col=cols
        )
      }
      
      else{
        segments(
          rep(diff(x.range)*bias.x[val]+diff(x.range)*line.stag,struct[val+1]),
          rep(bias.y*diff(y.range),struct[val+1]),
          rep(diff(x.range)*layer.x[val+1]-diff(x.range)*line.stag,struct[val+1]),
          get.ys(struct[val+1])[all.out],
          lwd=wts.rs[all.out],
          col=cols[all.out]
        )
      }
      
    }
  }
  
  #use functions to plot connections between layers
  #bias lines
  if(bias) bias.lines(bias.x,mod.in,nid=nid,rel.rsc=rel.rsc,all.out=all.out,pos.col=alpha(pos.col,alpha.val),
                      neg.col=alpha(neg.col,alpha.val))
  
  #layer lines, makes use of arguments to plot all or for individual layers
  #starts with input-hidden
  #uses 'all.in' argument to plot connection lines for all input nodes or a single node
  if(is.logical(all.in)){  
    mapply(
      function(x) layer.lines(mod.in,x,layer1=1,layer2=2,nid=nid,rel.rsc=rel.rsc,
        all.in=all.in,pos.col=alpha(pos.col,alpha.val),neg.col=alpha(neg.col,alpha.val)),
      1:struct[1]
    )
  }
  else{
    node.in<-which(x.names==all.in)
    layer.lines(mod.in,node.in,layer1=1,layer2=2,nid=nid,rel.rsc=rel.rsc,all.in=all.in,
                pos.col=alpha(pos.col,alpha.val),neg.col=alpha(neg.col,alpha.val))
  }
  #connections between hidden layers
  lays<-split(c(1,rep(2:(length(struct)-1),each=2),length(struct)),
              f=rep(1:(length(struct)-1),each=2))
  lays<-lays[-c(1,(length(struct)-1))]
  for(lay in lays){
    for(node in 1:struct[lay[1]]){
      layer.lines(mod.in,node,layer1=lay[1],layer2=lay[2],nid=nid,rel.rsc=rel.rsc,all.in=T,
                  pos.col=alpha(pos.col,alpha.val),neg.col=alpha(neg.col,alpha.val))
    }
  }
  #lines for hidden-output
  #uses 'all.out' argument to plot connection lines for all output nodes or a single node
  if(is.logical(all.out))
    mapply(
      function(x) layer.lines(mod.in,x,layer1=length(struct)-1,layer2=length(struct),out.layer=T,nid=nid,rel.rsc=rel.rsc,
                              all.in=all.in,pos.col=alpha(pos.col,alpha.val),neg.col=alpha(neg.col,alpha.val)),
      1:struct[length(struct)]
      )
  else{
    node.in<-which(y.names==all.out)
    layer.lines(mod.in,node.in,layer1=length(struct)-1,layer2=length(struct),out.layer=T,nid=nid,rel.rsc=rel.rsc,
                pos.col=pos.col,neg.col=neg.col,all.out=all.out)
  }
  
  #use functions to plot nodes
  for(i in 1:length(struct)){
    in.col<-circle.col
    layer.name<-'H'
    if(i==1) { layer.name<-'I'; in.col<-circle.col.inp}
    if(i==length(struct)) layer.name<-'O'
    layer.points(struct[i],layer.x[i],layer.name)
    }

  if(bias) bias.points(bias.x,bias.y,'B')
  
}


# ------------------------------------------------------------------------------------
# 번  호: 3
# 함수명: ts_graph_fn
# 입력값: input_df(그래프 작성을 위한 데이터), input.var(날짜 변수명), input.scales(축 범위 고정 혹은 변동)
# 설  명: ggplot을 이용한 시계열 그래프(line graph) 작성
# ------------------------------------------------------------------------------------

ts_graph_fn <- function(input_df, input.var = 'DATE', input.scales = c('free_x', 'free'), nrow = NULL, ncol = NULL){
      # 그래프 작성을 위한 데이터 변환
      sel_reshape_df <- reshape(data = input_df, idvar = input.var,
                                varying = names(input_df)[-1],
                                v.name = c('Value'),
                                times = names(input_df)[-1],
                                direction = 'long')
      names(sel_reshape_df) <- c('Date', 'Variable', 'Value')
      row.names(sel_reshape_df) <- NULL
      tail(sel_reshape_df)

      # ggplot을 이용한 시계열 그래프 작성
      date.var <- 'Date'
      var.x <- 'Value'
      group.var <- 'Variable'

      sel_reshape_df %>%
        ggplot() + 
        geom_line(aes_string(x = date.var, y = var.x)) +
        facet_wrap(facets = as.formula(paste('~', group.var)), scales = input.scales, labeller = label_both, nrow = nrow, ncol = ncol) +
        theme(strip.text.x = element_text(size = 16), axis.title = element_text(size = 16), axis.text = element_text(size = 12),
              axis.text.x = element_text(angle = 90, hjust = 0.5, vjust = 0.5)) +
        scale_x_datetime(date_breaks = '1 month', labels = date_format('%y-%m-%d %H:%M'))
}


# -------------------------------------------------------------------------------------------------
# 번  호: 4
# 함수명: restore_scale_fn
# 입력값: train_df(train 데이터프레임), y(종속변수), actual(예측 시 사용된 실제값), predictions(예측값)
# 설  명: 변경된 스케일을 원래 스케일로 복원
# -------------------------------------------------------------------------------------------------

restore_scale_fn <- function(train_df, y, actual, predictions){
    # 스케일을 원래 상태로 되돌리기
    y_mean <- mean(train_df[[y]], na.rm = T)
    y_sd <- sd(train_df[[y]], na.rm = T)
    
    predict_restore <- array()
    y_restore <- array()
    
    for (i in (1:length(predictions))){
        predict_restore[i] <- (predictions[i]*y_sd) + y_mean
        y_restore[i] <- (actual[i]*y_sd) + y_mean
        }

    # 결과 df 생성
    result_df <- data.frame(y_restore, predict_restore)
    names(result_df) <- c('actual', 'predicted')
    
    return (result_df)
}


# ------------------------------------------------------------------------------------
# 번  호: 5
# 함수명: train_plot_fn
# 입력값: y(종속변수), model(모델명), history(학습된 모델)
# 설  명: lstm, gru의 epoch별 loss, rmse확인, ann의 신경망 시각화
# ------------------------------------------------------------------------------------

train_plot_fn <- function(y, model, history){
    # 막 선택
    if (y=='SFLUX') membrane <- 'RO' else membrane <- 'MF'
    
    # 모델 시각화
    if (model=='nn'){
        # plot출력
        plot_nnet_fn(history)
        
        # plot 저장
        options(repr.plot.width = 10, repr.plot.height = 8)
        png(paste0('output/graph/', membrane, '_ann_visualization.png'))
        plot_nnet_fn(history)
        dev.off() 
        
    }else{  # lstm, gru일떄
        options(repr.plot.width = 20, repr.plot.height = 8)
        
        # plot 생성
        loss_df <- data.frame(epoch = 1:length(history$metrics$loss), loss = history$metrics$loss)
        loss_plot <- ggplot(data = loss_df, aes(x = epoch, y = loss)) +
                      geom_line(color = 'blue') +
                      geom_point() +
                      xlab('Epoch') +
                      ylab('Loss') +
                      ggtitle(paste0(membrane, '_', model, '_loss_plot')) + 
                      theme(plot.title = element_text(size = 20), 
                            axis.title.x = element_text(size = 15), 
                            axis.title.y = element_text(size = 15))
        
        rmse_df <- data.frame(epoch = 1:length(history$metrics$root_mean_squared_error), rmse = history$metrics$root_mean_squared_error)
        rmse_plot <- ggplot(data = rmse_df, aes(x = epoch, y = rmse)) +
                      geom_line(color = 'red') +
                      geom_point() +
                      xlab('Epoch') +
                      ylab('Rmse') +
                      ggtitle(paste0(membrane, '_', model, '_rmse_plot')) + 
                      theme(plot.title = element_text(size = 20), 
                            axis.title.x = element_text(size = 15), 
                            axis.title.y = element_text(size = 15))

        
        # plot 출력
        plot_layout <- grid.arrange(loss_plot, rmse_plot, nrow = 1)
        invisible(plot_layout)
        
        # plot 저장
        ggsave(paste0('./output/graph/', membrane, '_', model, '_loss_rmse_plot.png'), plot_layout, width = 10, height = 4, dpi = 100)
        }
}


# -----------------------------------------------------------------------------------------
# 번  호: 6
# 함수명: result_plot_fn
# 입력값: y(종속변수), df(실제값, 예측값이 들어있는 데이터프레임), div(train/test구분, 모델종류)
# 설  명:시계열 plot, 실제값, 예측값 비교 plot 생성, 저장
# # ---------------------------------------------------------------------------------------

# ggplot(data, aes(축)): 데이터 시각화하는 함수
# geom_line(): 선그래프 출력
result_plot_fn <- function(y, df, div){
    # 막 선택
    if (y=='SFLUX') membrane <- 'RO' else membrane <- 'MF'
    
    # plot 정의
    # 시계열 plot
    options(repr.plot.width = 25, repr.plot.height = 10)
    time_series_plot <- ggplot(data = df, aes(x = 1:nrow(df))) +  
                        geom_line(aes(y = actual, color = 'red'), size = 1) +
                        geom_line(aes(y = predicted, color = 'blue'), size = 1) +
                        ggtitle(paste0(membrane, '_', div, '_time_series_plot')) + 
                        theme(plot.title = element_text(size = 17), legend.text = element_text(size = 13), legend.position = c(0.87, 0.97))+
                        scale_color_manual(values = c('red' = 'red', 'blue' = 'blue'), labels = c('Actual', 'Predicted')) + 
                        labs(color = '')
    
    # 실제값, 예측값 비교 plot
    actual_prediction_plot <- ggplot(df, aes(x = predicted, y = actual)) + geom_point()+ 
                              geom_abline(slope = 1, intercept = 0, color='red') +
                              ggtitle(paste0(membrane, '_', div, '_actual_prediction_plot')) + 
                              theme(plot.title = element_text(size = 17)) +
                              xlab('Predicted') +
                              ylab('Actual')

    # plot 저장
    ggsave(paste0('./output/graph/', membrane, '_', div, '_time_series_plot.png'), time_series_plot, width = 5, height = 5, dpi = 100)
    ggsave(paste0('./output/graph/', membrane, '_', div, '_actual_prediction_plot.png'), actual_prediction_plot, width = 5, height = 5, dpi = 100)
   
    # plot 출력
    plot_layout <- grid.arrange(time_series_plot, actual_prediction_plot, nrow = 1)
    invisible(plot_layout)
}

                            
# -----------------------------------------------------------------------------------------
# 번  호: 7
# 함수명: evaluation_fn
# 입력값: y(종속변수), df(실제값, 예측값이 들어있는 데이터프레임), div(train/test구분, 모델종류)
# 설  명: 평가지표 생성 후 csv로 저장
# -----------------------------------------------------------------------------------------

# sqrt(): 제곱근 처리
# abs(): 절대값 반환
evaluation_fn <- function(y, df, div){
    # 막 선택
    if (y=='SFLUX') membrane <- 'RO' else membrane <- 'MF'
    
    # 평가지표 생성
    rmse <- sqrt(mean((df$actual-df$predicted)^2))
    mae <- mean(abs(df$actual-df$predicted))
    
    # df에 컬럼추가
    df$rmse <- rmse
    df$mae <- mae
    
    # 평가지표 저장
    write.csv(df, file = paste0('./output/', membrane, '_', div,'_evaluation.csv'), row.names = FALSE)

    # 평가지표 출력
    cat('rmse: ', rmse, '\n')
    cat('mae: ', mae)
}
